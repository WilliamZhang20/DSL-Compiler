#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/MatrixBuilder.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/LoopInfo.h>
#include <iostream>
#include <map>
#include <vector>
#include <cctype>

using namespace llvm;
using namespace llvm::orc;

// ──────────────────────────────────────────────────────────────
// Tokenizer (now supports mat4x4)
// ──────────────────────────────────────────────────────────────
enum Token {
    TokEof, TokFn, TokReturn, TokVar, TokIdent, TokNumber, TokFloat,
    TokLParen, TokRParen, TokLBrace, TokRBrace, TokLBracket, TokRBracket,
    TokColon, TokArrow, TokComma, TokDot, TokPlus, TokMul, TokSemi, TokFor
};

struct Tokenizer {
    const char* Ptr;
    std::string IdentStr;
    double FloatVal;
    Token CurTok;

    Tokenizer(const std::string& Input) : Ptr(Input.c_str()) { getNextToken(); }

    Token getNextToken() {
        while (*Ptr && std::isspace(*Ptr)) ++Ptr;
        if (!*Ptr) return CurTok = TokEof;

        if (std::isalpha(*Ptr)) {
            IdentStr = *Ptr++;
            while (std::isalnum(*Ptr) || *Ptr == '_') IdentStr += *Ptr++;
            if (IdentStr == "fn") return CurTok = TokFn;
            if (IdentStr == "return") return CurTok = TokReturn;
            if (IdentStr == "var") return CurTok = TokVar;
            if (IdentStr == "for") return CurTok = TokFor;
            return CurTok = TokIdent;
        }

        if (std::isdigit(*Ptr) || *Ptr == '.') {
            char* End;
            FloatVal = strtod(Ptr, &End);
            Ptr = End;
            return CurTok = TokFloat;
        }

        char c = *Ptr++;
        switch (c) {
            case '(': return CurTok = TokLParen;
            case ')': return CurTok = TokRParen;
            case '{': return CurTok = TokLBrace;
            case '}': return CurTok = TokRBrace;
            case '[': return CurTok = TokLBracket;
            case ']': return CurTok = TokRBracket;
            case ':': return CurTok = TokColon;
            case ',': return CurTok = TokComma;
            case '.': return CurTok = TokDot;
            case '+': return CurTok = TokPlus;
            case '*': return CurTok = TokMul;
            case ';': return CurTok = TokSemi;
            case '-':
                if (*Ptr == '>') { ++Ptr; return CurTok = TokArrow; }
                break;
        }
        std::cerr << "Unknown char: " << c << "\n";
        return CurTok = TokEof;
    }
};

// ──────────────────────────────────────────────────────────────
// Compiler with mat4x4 and custom matmul pass
// ──────────────────────────────────────────────────────────────
class Compiler {
    LLVMContext Context;
    std::unique_ptr<Module> Mod;
    IRBuilder<> Builder;
    std::unique_ptr<LLJIT> JIT;

    Type* FloatTy;
    VectorType* Vec4Ty;
    ArrayType* Mat4x4Ty;  // [4 x <4 x float>]

public:
    Compiler() : Builder(Context) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();

        FloatTy = Type::getFloatTy(Context);
        Vec4Ty = VectorType::get(FloatTy, 4, false);
        Mat4x4Ty = ArrayType::get(Vec4Ty, 4);

        Mod = std::make_unique<Module>("blaze_mat", Context);
        JIT = cantFail(LLJITBuilder().create());
    }

    Type* parseType(Tokenizer& Tok) {
        if (Tok.CurTok != TokIdent) return nullptr;
        if (Tok.IdentStr == "f32") { Tok.getNextToken(); return FloatTy; }
        if (Tok.IdentStr == "vec4") { Tok.getNextToken(); if (Tok.CurTok == TokIdent && Tok.IdentStr == "f32") Tok.getNextToken(); return Vec4Ty; }
        if (Tok.IdentStr == "mat4x4") { Tok.getNextToken(); if (Tok.CurTok == TokIdent && Tok.IdentStr == "f32") Tok.getNextToken(); return Mat4x4Ty; }
        return nullptr;
    }

    // Simple matrix load/store helpers
    Value* getRow(Value* Mat, Value* Idx) {
        return Builder.CreateExtractValue(Mat, (unsigned)cast<ConstantInt>(Idx)->getValue().getZExtValue());
    }

    void setRow(Value* Mat, Value* Idx, Value* Row) {
        Builder.CreateInsertValue(Mat, Row, (unsigned)cast<ConstantInt>(Idx)->getValue().getZExtValue());
    }

    // Parse simple expressions (enough for matmul)
    Value* parsePrimary(Tokenizer& Tok, std::map<std::string, Value*>& Locals) {
        if (Tok.CurTok == TokFloat) {
            Value* V = ConstantFP::get(FloatTy, Tok.FloatVal);
            Tok.getNextToken();
            return V;
        }
        if (Tok.CurTok == TokIdent) {
            std::string Name = Tok.IdentStr;
            Tok.getNextToken();
            Value* V = Locals[Name];
            if (!V) { std::cerr << "unknown var\n"; return nullptr; }

            // A[i] for row access
            if (Tok.CurTok == TokLBracket) {
                Tok.getNextToken();
                Value* Idx = parsePrimary(Tok, Locals);
                if (!Idx || !isa<ConstantInt>(Idx)) { std::cerr << "const index only\n"; return nullptr; }
                Tok.getNextToken(); // ]
                V = Builder.CreateExtractValue(V, (unsigned)cast<ConstantInt>(Idx)->getZExtValue());
            }
            return V;
        }
        return nullptr;
    }

    Value* parseMul(Tokenizer& Tok, std::map<std::string, Value*>& Locals) {
        Value* L = parsePrimary(Tok, Locals);
        while (Tok.CurTok == TokMul) {
            Tok.getNextToken();
            Value* R = parsePrimary(Tok, Locals);
            L = Builder.CreateFMul(L, R);
        }
        return L;
    }

    Value* parseAdd(Tokenizer& Tok, std::map<std::string, Value*>& Locals) {
        Value* L = parseMul(Tok, Locals);
        while (Tok.CurTok == TokPlus) {
            Tok.getNextToken();
            Value* R = parseMul(Tok, Locals);
            L = Builder.CreateFAdd(L, R);
        }
        return L;
    }

    void compile(const std::string& Source) {
        Mod = std::make_unique<Module>("blaze_mat", Context);
        Tokenizer Tok(Source);

        while (Tok.CurTok == TokFn) {
            Tok.getNextToken();
            std::string FnName = Tok.IdentStr;
            Tok.getNextToken(); // name consumed

            if (Tok.CurTok != TokLParen) { std::cerr << "(\n"; return; }
            std::vector<std::pair<std::string, Type*>> Args;
            Tok.getNextToken();
            while (Tok.CurTok == TokIdent) {
                std::string Name = Tok.IdentStr;
                Tok.getNextToken();
                if (Tok.CurTok != TokColon) return;
                Tok.getNextToken();
                Type* Ty = parseType(Tok);
                if (!Ty) return;
                Args.emplace_back(Name, Ty);
                if (Tok.CurTok == TokComma) Tok.getNextToken();
                else break;
            }
            if (Tok.CurTok != TokRParen) return;
            Tok.getNextToken();

            Type* RetTy = Mat4x4Ty;
            if (Tok.CurTok == TokArrow) {
                Tok.getNextToken();
                RetTy = parseType(Tok);
            }

            if (Tok.CurTok != TokLBrace) return;
            Tok.getNextToken();

            std::vector<Type*> ArgTys;
            for (auto &A : Args)
                ArgTys.push_back(A.second);

            Function* F = Function::Create(
                FunctionType::get(RetTy, ArgTys, false),
                GlobalValue::ExternalLinkage,
                FnName,
                Mod.get()
            );

            BasicBlock* Entry = BasicBlock::Create(Context, "entry", F);
            Builder.SetInsertPoint(Entry);

            std::map<std::string, Value*> Locals;
            unsigned i = 0;
            for (Argument &A : F->args()) {
                A.setName(Args[i].first);
                Locals[Args[i++].first] = &A;
            }

            // Very simple body parsing — only naïve matmul for now
            Value* C = Builder.CreateAlloca(Mat4x4Ty);
            C = Builder.CreateLoad(Mat4x4Ty, C);

            // Zero init C
            for (int i = 0; i < 4; ++i)
                C = Builder.CreateInsertValue(C, ConstantAggregateZero::get(Vec4Ty), i);

            for (int i = 0; i < 4; ++i) {
                for (int j = 0; j < 4; ++j) {
                    Value* sum = ConstantFP::get(FloatTy, 0.0);
                    for (int k = 0; k < 4; ++k) {
                        Value* a = Builder.CreateExtractValue(
                            Locals["A"],
                            { static_cast<unsigned>(i), static_cast<unsigned>(k) }
                        );

                        Value* b = Builder.CreateExtractValue(
                            Locals["B"],
                            { static_cast<unsigned>(k), static_cast<unsigned>(j) }
                        );
                        sum = Builder.CreateFAdd(sum, Builder.CreateFMul(a, b));
                    }
                    Value* row = Builder.CreateExtractValue(C, i);
                    row = Builder.CreateInsertElement(row, sum, Builder.getInt32(j));
                    C = Builder.CreateInsertValue(C, row, i);
                }
            }

            Builder.CreateRet(C);

            verifyFunction(*F);
            Tok.getNextToken(); // skip }
        }

        // ───── Custom Pass: Replace naïve matmul with llvm.matrix.multiply ─────
        legacy::FunctionPassManager FPM(Mod.get());

        for (Function &F : *Mod) {
            if (F.getName() != "matmul") continue;

            // Very simple pattern match: look for triple nested loops (we know we emitted them)
            // In real life you'd write a proper pattern matcher — here we just directly emit the intrinsic
            F.deleteBody();
            BasicBlock* BB = BasicBlock::Create(Context, "entry", &F);
            Builder.SetInsertPoint(BB);

            auto Args = F.arg_begin();
            Value* A = &*Args++;
            Value* B = &*Args++;

            // @llvm.matrix.multiply.f32 (matA, matB, 4, 4, 4) -> row-major, row-major
            Function* Intr = Intrinsic::getDeclaration(Mod.get(), Intrinsic::matrix_multiply,
                                                      {Mat4x4Ty, Mat4x4Ty});
            Value* Result = Builder.CreateCall(Intr, {A, B,
                Builder.getInt32(4), Builder.getInt32(4), Builder.getInt32(4)});

            Builder.CreateRet(Result);
        }

        Mod->print(errs(), nullptr);  // See the optimized IR!

        // Run standard opts too
        legacy::PassManager MPM;
        MPM.add(createVerifierPass());
        MPM.run(*Mod);

        cantFail(JIT->addIRModule(ThreadSafeModule(std::move(Mod),
            ThreadSafeContext(std::make_unique<LLVMContext>()))));

        auto MainSym = JIT->lookup("main");
        if (MainSym) {
            auto Addr = *MainSym; // ExecutorAddr
            using MainTy = void (*)();
            MainTy MainPtr = Addr.toPtr<MainTy>();
            MainPtr();
        }
    }
};

int main() {
    Compiler C;
    std::cout << "Blaze REPL v4 — mat4x4 + matrix.multiply intrinsic!\n";

    std::string Line, Source;
    while (std::cout << "> " && std::getline(std::cin, Line)) {
        if (Line.empty() && !Source.empty()) {
            C.compile(Source);
            Source.clear();
        } else if (!Line.empty()) {
            Source += Line + "\n";
        }
    }
}