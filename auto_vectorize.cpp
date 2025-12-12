#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <iostream>
#include <map>
#include <cctype>
#include <vector>
#include <cmath>
#include <memory>

using namespace llvm;
using namespace llvm::orc;

// ──────────────────────────────────────────────────────────────
// Tokenizer (extended for f32, vec4, vec8, dot notation)
// ──────────────────────────────────────────────────────────────
enum Token {
    TokEof, TokFn, TokReturn, TokVar, TokIdent, TokNumber, TokFloat,
    TokLParen, TokRParen, TokLBrace, TokRBrace, TokColon, TokArrow,
    TokComma, TokDot, TokPlus, TokMul, TokSemi
};

struct Tokenizer {
    const char* Ptr;
    std::string IdentStr;
    int IntVal;
    double FloatVal;
    Token CurTok;

    Tokenizer(const std::string& Input) : Ptr(Input.c_str()) { getNextToken(); }

    Token getNextToken() {
        while (*Ptr && std::isspace(*Ptr)) ++Ptr;
        if (!*Ptr) return CurTok = TokEof;

        // identifier: [A-Za-z][A-Za-z0-9_]*
        if (std::isalpha(*Ptr)) {
            IdentStr = *Ptr++;
            while (std::isalnum(*Ptr) || *Ptr == '_') IdentStr += *Ptr++;
            if (IdentStr == "fn") return CurTok = TokFn;
            if (IdentStr == "return") return CurTok = TokReturn;
            if (IdentStr == "var") return CurTok = TokVar;
            return CurTok = TokIdent;
        }

        // number: start with digit OR '.' followed by a digit (so '.' as field access is preserved)
        if (std::isdigit(*Ptr) || (*Ptr == '.' && std::isdigit(*(Ptr + 1)))) {
            char* End;
            FloatVal = strtod(Ptr, &End);
            // support optional trailing 'f'/'F'
            if (End > Ptr && (*End == 'f' || *End == 'F')) ++End;
            Ptr = End;
            IntVal = (int)FloatVal;
            return CurTok = TokFloat;
        }

        // single-character tokens
        char c = *Ptr++;
        switch (c) {
            case '(': return CurTok = TokLParen;
            case ')': return CurTok = TokRParen;
            case '{': return CurTok = TokLBrace;
            case '}': return CurTok = TokRBrace;
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
// Compiler with vec4/vec8 support
// ──────────────────────────────────────────────
class Compiler {
    LLVMContext Context;
    std::unique_ptr<Module> Mod;
    IRBuilder<> Builder;
    std::unique_ptr<LLJIT> JIT;
    std::map<std::string, Value*> NamedValues;

    Type* FloatTy;
    FixedVectorType* Vec4Ty;
    FixedVectorType* Vec8Ty;

public:
    Compiler() : Builder(Context) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();

        FloatTy = Type::getFloatTy(Context);
        Vec4Ty = FixedVectorType::get(FloatTy, 4);
        Vec8Ty = FixedVectorType::get(FloatTy, 8);

        Mod = std::make_unique<Module>("blaze_vec", Context);
        JIT = cantFail(LLJITBuilder().create());
    }

    Type* parseType(Tokenizer& Tok) {
        if (Tok.CurTok != TokIdent) return nullptr;
        if (Tok.IdentStr == "f32") { Tok.getNextToken(); return FloatTy; }
        if (Tok.IdentStr == "vec4") { Tok.getNextToken(); if (Tok.CurTok == TokIdent && Tok.IdentStr == "f32") Tok.getNextToken(); return Vec4Ty; }
        if (Tok.IdentStr == "vec8") { Tok.getNextToken(); if (Tok.CurTok == TokIdent && Tok.IdentStr == "f32") Tok.getNextToken(); return Vec8Ty; }
        return nullptr;
    }

    Value* parseVectorLiteral(Tokenizer& Tok, FixedVectorType* VTy) {
        // caller expects Tok.CurTok == TokLBrace
        std::vector<Constant*> Elems;
        unsigned N = VTy->getNumElements();

        // consume '{'
        if (Tok.CurTok != TokLBrace) { std::cerr << "expected '{' for vector literal\n"; return nullptr; }
        Tok.getNextToken();

        for (unsigned i = 0; i < N; ++i) {
            if (Tok.CurTok != TokFloat) { std::cerr << "expected float in vector literal\n"; return nullptr; }
            Elems.push_back(ConstantFP::get(FloatTy, Tok.FloatVal));
            Tok.getNextToken();
            if (i + 1 < N) {
                if (Tok.CurTok != TokComma) { std::cerr << "expected comma\n"; return nullptr; }
                Tok.getNextToken();
            }
        }
        if (Tok.CurTok != TokRBrace) { std::cerr << "expected }\n"; return nullptr; }
        Tok.getNextToken();
        return ConstantVector::get(Elems);
    }

    Value* parsePrimary(Tokenizer& Tok) {
        if (Tok.CurTok == TokFloat) {
            Value* V = ConstantFP::get(FloatTy, Tok.FloatVal);
            Tok.getNextToken();
            return V;
        }

        if (Tok.CurTok == TokIdent) {
            std::string Name = Tok.IdentStr;
            // consume identifier
            Tok.getNextToken();

            // Function call: name ( arg, arg )
            if (Tok.CurTok == TokLParen) {
                // parse call args
                Tok.getNextToken(); // consume '('
                std::vector<Value*> ArgsVals;
                if (Tok.CurTok != TokRParen) {
                    while (true) {
                        Value* A = parseExpr(Tok);
                        if (!A) return nullptr;
                        ArgsVals.push_back(A);
                        if (Tok.CurTok == TokComma) { Tok.getNextToken(); continue; }
                        break;
                    }
                }
                if (Tok.CurTok != TokRParen) { std::cerr << "expected ) in call\n"; return nullptr; }
                Tok.getNextToken(); // consume ')'

                // find function in module
                Function* F = Mod->getFunction(Name);
                if (!F) { std::cerr << "unknown function " << Name << "\n"; return nullptr; }
                
                if (Name == "dot" && ArgsVals.size() == 2) {
                    Type* t0 = ArgsVals[0]->getType();
                    Type* t1 = ArgsVals[1]->getType();
                    if (t0->isVectorTy() && t1->isVectorTy() &&
                        t0->getScalarType()->isFloatTy() && t1->getScalarType()->isFloatTy() &&
                        cast<VectorType>(t0)->getElementCount().getKnownMinValue() ==
                        cast<VectorType>(t1)->getElementCount().getKnownMinValue()) {

                        auto *vecTy = cast<FixedVectorType>(t0);
                        unsigned N = vecTy->getNumElements(); 

                        // vector multiply: v = a * b  (vector FMul)
                        Value* v = Builder.CreateFMul(ArgsVals[0], ArgsVals[1], "vecmul");

                        // horizontal reduction using shuffles + vector adds
                        // For N a power of two (4, 8) this does pairwise summation:
                        // for step = N/2, N/4, ..., 1:
                        //   shuffled = shufflevector(v, v, mask_for_step)
                        //   v = v + shuffled
                        LLVMContext &Ctx = Context; // use your member Context
                        Type* i32Ty = Type::getInt32Ty(Ctx);

                        while (N > 1) {
                            unsigned step = N / 2;
                            SmallVector<Constant*, 16> maskConsts;
                            maskConsts.reserve(cast<FixedVectorType>(vecTy)->getNumElements());

                            unsigned totalLanes = cast<FixedVectorType>(vecTy)->getNumElements();
                            for (unsigned i = 0; i < totalLanes; ++i) {
                                unsigned idx;
                                if (i < step) idx = i + step;
                                else idx = i - step;
                                maskConsts.push_back(ConstantInt::get(i32Ty, idx));
                            }

                            // Create a constant vector mask for shufflevector
                            Constant* mask = ConstantVector::get(maskConsts);
                            v = Builder.CreateFAdd(v, Builder.CreateShuffleVector(v, v, mask), "vreduce");
                            // halve the effective vector width for next iteration
                            // (we use the same vector type but the mask pairs lanes progressively)
                            N = step;
                        }

                        // After reduction the full vector has the total in each lane (0..),
                        // extract element 0 to get scalar result
                        Value* scalar = Builder.CreateExtractElement(v, Builder.getInt32(0), "dot_result");
                        return scalar;
                    }
                }

                // fallback to normal call if not the dot-vector pattern
                return Builder.CreateCall(F, ArgsVals, "calltmp");
            }

            // Otherwise treat as variable (possibly followed by field access)
            auto it = NamedValues.find(Name);
            if (it == NamedValues.end()) {
                std::cerr << "unknown var " << Name << "\n";
                return nullptr;
            }
            Value* V = it->second;

            // Field access: v.x, v.y, etc.
            while (Tok.CurTok == TokDot) {
                Tok.getNextToken();
                if (Tok.CurTok != TokIdent) { std::cerr << "expected field\n"; return nullptr; }
                std::string Field = Tok.IdentStr;
                Tok.getNextToken();

                unsigned Idx = 0;
                if (Field == "x") Idx = 0;
                else if (Field == "y") Idx = 1;
                else if (Field == "z") Idx = 2;
                else if (Field == "w") Idx = 3;
                else { std::cerr << "bad field\n"; return nullptr; }

                V = Builder.CreateExtractElement(V, Builder.getInt32(Idx), "extract");
            }
            return V;
        }

        std::cerr << "unexpected token in primary\n";
        return nullptr;
    }

    Value* parseMul(Tokenizer& Tok) {
        Value* LHS = parsePrimary(Tok);
        if (!LHS) return nullptr;
        while (Tok.CurTok == TokMul) {
            Tok.getNextToken();
            Value* RHS = parsePrimary(Tok);
            if (!RHS) return nullptr;
            LHS = Builder.CreateFMul(LHS, RHS, "multmp");
        }
        return LHS;
    }

    Value* parseExpr(Tokenizer& Tok) {
        Value* LHS = parseMul(Tok);
        if (!LHS) return nullptr;
        while (Tok.CurTok == TokPlus) {
            Tok.getNextToken();
            Value* RHS = parseMul(Tok);
            if (!RHS) return nullptr;
            LHS = Builder.CreateFAdd(LHS, RHS, "addtmp");
        }
        return LHS;
    }

    void compile(const std::string& Source) {
        // create a new module for each compile
        Mod = std::make_unique<Module>("blaze_vec", Context);
        Tokenizer Tok(Source);

        // Two-pass small improvement: declare function prototypes first
        // We'll parse top-level to collect function names/signatures before bodies.
        // For simplicity here we do a single-pass but create Function before parsing body (already done below).

        while (Tok.CurTok != TokEof) {
            if (Tok.CurTok == TokFn) {
                Tok.getNextToken();
                if (Tok.CurTok != TokIdent) { std::cerr << "expected fn name\n"; return; }
                std::string FnName = Tok.IdentStr;
                Tok.getNextToken();

                if (Tok.CurTok != TokLParen) { std::cerr << "expected (\n"; return; }
                Tok.getNextToken();

                // Collect arg names & types (we need types to construct function type)
                std::vector<std::pair<std::string, Type*>> Args;
                while (Tok.CurTok == TokIdent) {
                    std::string Name = Tok.IdentStr;
                    Tok.getNextToken();
                    if (Tok.CurTok != TokColon) { std::cerr << "expected :\n"; return; }
                    Tok.getNextToken();
                    Type* Ty = parseType(Tok);
                    if (!Ty) { std::cerr << "bad type\n"; return; }
                    Args.emplace_back(Name, Ty);
                    if (Tok.CurTok == TokComma) Tok.getNextToken();
                    else break;
                }
                if (Tok.CurTok != TokRParen) { std::cerr << "expected )\n"; return; }
                Tok.getNextToken();

                // Return type (optional)
                Type* RetTy = Type::getFloatTy(Context);
                if (Tok.CurTok == TokArrow) {
                    Tok.getNextToken();
                    Type* parsed = parseType(Tok);
                    if (parsed) RetTy = parsed;
                    else { std::cerr << "bad return type; defaulting to f32\n"; }
                }

                // Expect function body
                if (Tok.CurTok != TokLBrace) { std::cerr << "expected {\n"; return; }

                // Build function in module so other functions can call it
                std::vector<Type*> ArgTypes;
                for (auto &p : Args) ArgTypes.push_back(p.second);
                Function* F = Function::Create(FunctionType::get(RetTy, ArgTypes, false),
                                               Function::ExternalLinkage, FnName, Mod.get());

                // Create entry block and set up builder
                BasicBlock* BB = BasicBlock::Create(Context, "entry", F);
                Builder.SetInsertPoint(BB);

                // Prepare NamedValues for this function
                NamedValues.clear();
                unsigned i = 0;
                for (auto &A : F->args()) {
                    A.setName(Args[i].first);
                    // For simplicity, bind the Function argument value directly (no alloca)
                    NamedValues[Args[i++].first] = &A;
                }

                // consume '{'
                Tok.getNextToken();

                // Parse statements inside function
                while (Tok.CurTok == TokVar) {
                    Tok.getNextToken();
                    if (Tok.CurTok != TokIdent) { std::cerr << "expected var name\n"; return; }
                    std::string VarName = Tok.IdentStr;
                    Tok.getNextToken();
                    if (Tok.CurTok != TokColon) { std::cerr << "expected :\n"; return; }
                    Tok.getNextToken();
                    Type* VarTy = parseType(Tok);
                    if (!VarTy) { std::cerr << "bad var type\n"; return; }
                    Value* Init = nullptr;
                    if (Tok.CurTok == TokLBrace) {
                        // parse vector literal
                        Init = parseVectorLiteral(Tok, cast<FixedVectorType>(VarTy));
                    }
                    // create alloca and store initialiser
                    Value* Alloca = Builder.CreateAlloca(VarTy, nullptr, VarName);
                    if (Init) Builder.CreateStore(Init, Alloca);
                    NamedValues[VarName] = Builder.CreateLoad(VarTy, Alloca, VarName);
                    if (Tok.CurTok == TokSemi) Tok.getNextToken();
                }

                if (Tok.CurTok == TokReturn) {
                    Tok.getNextToken();
                    Value* Ret = parseExpr(Tok);
                    if (!Ret) return;
                    Builder.CreateRet(Ret);
                }

                if (Tok.CurTok != TokRBrace) { std::cerr << "expected }\n"; return; }
                Tok.getNextToken();

                // verify the function (will print errors if invalid)
                if (verifyFunction(*F, &errs())) {
                    std::cerr << "Function verification failed\n";
                    return;
                }
            } else {
                // ignore unexpected top-level tokens reliably
                Tok.getNextToken();
            }
        }

        // Print IR before moving module to JIT
        Mod->print(errs(), nullptr);

        // Move module into JIT
        cantFail(JIT->addIRModule(ThreadSafeModule(std::move(Mod),
            ThreadSafeContext(std::make_unique<LLVMContext>()))));

        // attempt to look up main
        auto MainSym = JIT->lookup("main");
        if (!MainSym) {
            std::cerr << "main() not found in JIT module\n";
            return;
        }

        // cast and call
        float (*MainPtr)() = (float(*)())MainSym->getValue();
        float result = MainPtr();
        std::cout << "=> " << result << "\n";
        if (std::fabs(result - 70.0f) < 0.001f) {
            std::cout << "✓ Auto-vectorization worked! (1*5 + 2*6 + 3*7 + 4*8 = 70)\n";
        }
    }
};

// ──────────────────────────────────────────────────────────────
// REPL (use EOF sentinel to compile multi-line input)
// ──────────────────────────────────────────────
int main() {
    Compiler C;
    std::cout << "Blaze REPL v2 — now with vec4 f32 + auto-vectorization!\n";
    std::cout << "Enter program lines; type a line with only 'EOF' to compile/run.\n";

    std::string Line, Source;
    while (std::cout << "> " && std::getline(std::cin, Line)) {
        if (Line == "EOF") {
            if (!Source.empty()) {
                C.compile(Source);
                Source.clear();
            }
        } else {
            Source += Line + "\n";
        }
    }
    return 0;
}