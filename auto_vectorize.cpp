#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IR/Intrinsics.h>
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
// Tokenizer (extended for control flow)
// ──────────────────────────────────────────────────────────────
enum Token {
    TokEof, TokFn, TokReturn, TokVar, TokIdent, TokNumber, TokFloat,
    TokLParen, TokRParen, TokLBrace, TokRBrace, TokColon, TokArrow,
    TokComma, TokDot, TokPlus, TokMul, TokSemi, TokIf, TokElse,
    TokFor, TokIn, TokRange, TokLess, TokGreater, TokEqual, TokAssign
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
            if (IdentStr == "if") return CurTok = TokIf;
            if (IdentStr == "else") return CurTok = TokElse;
            if (IdentStr == "for") return CurTok = TokFor;
            if (IdentStr == "in") return CurTok = TokIn;
            if (IdentStr == "range") return CurTok = TokRange;
            return CurTok = TokIdent;
        }

        // number
        if (std::isdigit(*Ptr) || (*Ptr == '.' && std::isdigit(*(Ptr + 1)))) {
            char* End;
            FloatVal = strtod(Ptr, &End);
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
            case '<': return CurTok = TokLess;
            case '>': return CurTok = TokGreater;
            case '-':
                if (*Ptr == '>') { ++Ptr; return CurTok = TokArrow; }
                break;
            case '=':
                if (*Ptr == '=') { ++Ptr; return CurTok = TokEqual; }
                return CurTok = TokAssign;
        }
        std::cerr << "Unknown char: " << c << "\n";
        return CurTok = TokEof;
    }
};

// ──────────────────────────────────────────────────────────────
// Compiler with vec4/vec8 support + swizzling + control flow
// ──────────────────────────────────────────────────────────────
class Compiler {
    LLVMContext Context;
    std::unique_ptr<Module> Mod;
    IRBuilder<> Builder;
    std::unique_ptr<LLJIT> JIT;
    std::map<std::string, Value*> NamedValues;
    std::map<std::string, AllocaInst*> NamedAllocas; // For mutable variables (when needed)

    Type* FloatTy;
    Type* Int32Ty;
    FixedVectorType* Vec4Ty;
    FixedVectorType* Vec8Ty;

public:
    Compiler() : Builder(Context) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();

        FloatTy = Type::getFloatTy(Context);
        Int32Ty = Type::getInt32Ty(Context);
        Vec4Ty = FixedVectorType::get(FloatTy, 4);
        Vec8Ty = FixedVectorType::get(FloatTy, 8);

        Mod = std::make_unique<Module>("blaze_vec", Context);
        JIT = cantFail(LLJITBuilder().create());
    }

    Type* parseType(Tokenizer& Tok) {
        if (Tok.CurTok != TokIdent) return nullptr;
        if (Tok.IdentStr == "f32") { Tok.getNextToken(); return FloatTy; }
        if (Tok.IdentStr == "i32") { Tok.getNextToken(); return Int32Ty; }
        if (Tok.IdentStr == "vec4") { 
            Tok.getNextToken(); 
            if (Tok.CurTok == TokIdent && Tok.IdentStr == "f32") Tok.getNextToken(); 
            return Vec4Ty; 
        }
        if (Tok.IdentStr == "vec8") { 
            Tok.getNextToken(); 
            if (Tok.CurTok == TokIdent && Tok.IdentStr == "f32") Tok.getNextToken(); 
            return Vec8Ty; 
        }
        return nullptr;
    }

    Value* parseVectorLiteral(Tokenizer& Tok, FixedVectorType* VTy) {
        std::vector<Constant*> Elems;
        unsigned N = VTy->getNumElements();

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

    // Helper: reduce vector using LLVM intrinsic
    Value* reduceVector(Value* V) {
        auto *VecTy = cast<VectorType>(V->getType());
        Function *Reduce = Intrinsic::getDeclaration(
            Mod.get(),
            Intrinsic::vector_reduce_fadd,
            VecTy
        );
        Value *Zero = ConstantFP::get(FloatTy, 0.0);
        return Builder.CreateCall(Reduce, {Zero, V}, "dot");
    }

    Value* parseIfExpr(Tokenizer& Tok) {
        // if <expr> { <expr> } else { <expr> }
        Tok.getNextToken(); // consume 'if'
        
        Value* CondV = parseExpr(Tok);
        if (!CondV) return nullptr;

        // Convert condition to bool by comparing with 0.0
        Value* Zero = ConstantFP::get(FloatTy, 0.0);
        Value* Cond = Builder.CreateFCmpONE(CondV, Zero, "ifcond");

        Function* F = Builder.GetInsertBlock()->getParent();
        BasicBlock* ThenBB = BasicBlock::Create(Context, "then", F);
        BasicBlock* ElseBB = BasicBlock::Create(Context, "else", F);
        BasicBlock* MergeBB = BasicBlock::Create(Context, "merge", F);

        Builder.CreateCondBr(Cond, ThenBB, ElseBB);

        // THEN branch
        Builder.SetInsertPoint(ThenBB);
        if (Tok.CurTok != TokLBrace) { std::cerr << "expected '{' after if condition\n"; return nullptr; }
        Tok.getNextToken();
        Value* ThenV = parseExpr(Tok);
        if (!ThenV) return nullptr;
        if (Tok.CurTok != TokRBrace) { std::cerr << "expected '}' after then block\n"; return nullptr; }
        Tok.getNextToken();
        Builder.CreateBr(MergeBB);
        ThenBB = Builder.GetInsertBlock(); // Update in case of nested blocks

        // ELSE branch
        Builder.SetInsertPoint(ElseBB);
        if (Tok.CurTok != TokElse) { std::cerr << "expected 'else'\n"; return nullptr; }
        Tok.getNextToken();
        if (Tok.CurTok != TokLBrace) { std::cerr << "expected '{' after else\n"; return nullptr; }
        Tok.getNextToken();
        Value* ElseV = parseExpr(Tok);
        if (!ElseV) return nullptr;
        if (Tok.CurTok != TokRBrace) { std::cerr << "expected '}' after else block\n"; return nullptr; }
        Tok.getNextToken();
        Builder.CreateBr(MergeBB);
        ElseBB = Builder.GetInsertBlock();

        // MERGE
        Builder.SetInsertPoint(MergeBB);
        PHINode* Phi = Builder.CreatePHI(ThenV->getType(), 2, "ifphi");
        Phi->addIncoming(ThenV, ThenBB);
        Phi->addIncoming(ElseV, ElseBB);

        return Phi;
    }

    Value* parseForLoop(Tokenizer& Tok) {
        // for <var> in range(<start>, <end>) { <body> }
        Tok.getNextToken(); // consume 'for'

        if (Tok.CurTok != TokIdent) { std::cerr << "expected loop variable\n"; return nullptr; }
        std::string VarName = Tok.IdentStr;
        Tok.getNextToken();

        if (Tok.CurTok != TokIn) { std::cerr << "expected 'in'\n"; return nullptr; }
        Tok.getNextToken();

        if (Tok.CurTok != TokIdent || Tok.IdentStr != "range") { 
            std::cerr << "expected 'range'\n"; return nullptr; 
        }
        Tok.getNextToken();

        if (Tok.CurTok != TokLParen) { std::cerr << "expected '('\n"; return nullptr; }
        Tok.getNextToken();

        Value* StartV = parseExpr(Tok);
        if (!StartV) return nullptr;

        if (Tok.CurTok != TokComma) { std::cerr << "expected ','\n"; return nullptr; }
        Tok.getNextToken();

        Value* EndV = parseExpr(Tok);
        if (!EndV) return nullptr;

        if (Tok.CurTok != TokRParen) { std::cerr << "expected ')'\n"; return nullptr; }
        Tok.getNextToken();

        if (Tok.CurTok != TokLBrace) { std::cerr << "expected '{'\n"; return nullptr; }
        Tok.getNextToken();

        // Create loop structure
        Function* F = Builder.GetInsertBlock()->getParent();
        BasicBlock* PreheaderBB = Builder.GetInsertBlock();
        BasicBlock* LoopBB = BasicBlock::Create(Context, "loop", F);
        BasicBlock* AfterBB = BasicBlock::Create(Context, "afterloop", F);

        // Convert start/end to int32
        Value* Start = Builder.CreateFPToSI(StartV, Int32Ty, "start_int");
        Value* End = Builder.CreateFPToSI(EndV, Int32Ty, "end_int");

        Builder.CreateBr(LoopBB);

        // Loop body
        Builder.SetInsertPoint(LoopBB);
        PHINode* IndVar = Builder.CreatePHI(Int32Ty, 2, VarName);
        IndVar->addIncoming(Start, PreheaderBB);

        // Store old binding and create new one
        Value* OldVal = NamedValues[VarName];
        Value* LoopVar = Builder.CreateSIToFP(IndVar, FloatTy, VarName + "_f");
        NamedValues[VarName] = LoopVar;

        // Parse loop body
        Value* BodyV = parseExpr(Tok);
        if (!BodyV) return nullptr;

        if (Tok.CurTok != TokRBrace) { std::cerr << "expected '}' after loop body\n"; return nullptr; }
        Tok.getNextToken();

        // Increment and check
        Value* StepVal = ConstantInt::get(Int32Ty, 1);
        Value* NextVar = Builder.CreateAdd(IndVar, StepVal, "nextvar");
        Value* EndCond = Builder.CreateICmpSLT(NextVar, End, "loopcond");

        BasicBlock* LoopEndBB = Builder.GetInsertBlock();
        Builder.CreateCondBr(EndCond, LoopBB, AfterBB);

        // For loops return 0.0
        Builder.SetInsertPoint(AfterBB);

        IndVar->addIncoming(NextVar, LoopEndBB);

        // Restore old binding
        if (OldVal) NamedValues[VarName] = OldVal;
        else NamedValues.erase(VarName);

        return ConstantFP::get(FloatTy, 0.0);
    }

    Value* parsePrimary(Tokenizer& Tok) {
        if (Tok.CurTok == TokIf) {
            return parseIfExpr(Tok);
        }

        if (Tok.CurTok == TokFor) {
            return parseForLoop(Tok);
        }

        if (Tok.CurTok == TokFloat) {
            Value* V = ConstantFP::get(FloatTy, Tok.FloatVal);
            Tok.getNextToken();
            return V;
        }

        if (Tok.CurTok == TokIdent) {
            std::string Name = Tok.IdentStr;
            Tok.getNextToken();

            // Function call
            if (Tok.CurTok == TokLParen) {
                Tok.getNextToken();
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
                Tok.getNextToken();

                Function* F = Mod->getFunction(Name);
                if (!F) { std::cerr << "unknown function " << Name << "\n"; return nullptr; }
                
                // Optimized dot product using vector_reduce_fadd intrinsic
                if (Name == "dot" && ArgsVals.size() == 2) {
                    Type* t0 = ArgsVals[0]->getType();
                    Type* t1 = ArgsVals[1]->getType();
                    if (t0->isVectorTy() && t1->isVectorTy() &&
                        t0->getScalarType()->isFloatTy() && t1->getScalarType()->isFloatTy() &&
                        cast<VectorType>(t0)->getElementCount().getKnownMinValue() ==
                        cast<VectorType>(t1)->getElementCount().getKnownMinValue()) {
                        
                        Value* mul = Builder.CreateFMul(ArgsVals[0], ArgsVals[1], "vecmul");
                        return reduceVector(mul);
                    }
                }

                return Builder.CreateCall(F, ArgsVals, "calltmp");
            }

            // Variable
            auto it = NamedValues.find(Name);
            if (it == NamedValues.end()) {
                std::cerr << "unknown var " << Name << "\n";
                return nullptr;
            }
            Value* V = it->second;

            // Swizzling and field access: v.x, v.yx, v.xyz, v.zwxy, etc.
            while (Tok.CurTok == TokDot) {
                Tok.getNextToken();
                if (Tok.CurTok != TokIdent) { std::cerr << "expected field or swizzle\n"; return nullptr; }
                std::string Swizzle = Tok.IdentStr;
                Tok.getNextToken();

                if (!V->getType()->isVectorTy()) {
                    std::cerr << "cannot swizzle non-vector type\n";
                    return nullptr;
                }

                // Parse swizzle string
                SmallVector<Constant*, 8> Mask;
                for (char c : Swizzle) {
                    int idx = -1;
                    if (c == 'x') idx = 0;
                    else if (c == 'y') idx = 1;
                    else if (c == 'z') idx = 2;
                    else if (c == 'w') idx = 3;
                    else { std::cerr << "invalid swizzle component: " << c << "\n"; return nullptr; }
                    Mask.push_back(Builder.getInt32(idx));
                }

                if (Mask.size() == 1) {
                    // Single element extract
                    V = Builder.CreateExtractElement(V, Mask[0], "extract");
                } else {
                    // Multi-element swizzle
                    V = Builder.CreateShuffleVector(V, V, ConstantVector::get(Mask), "swizzle");
                }
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
        Mod = std::make_unique<Module>("blaze_vec", Context);
        Tokenizer Tok(Source);

        while (Tok.CurTok != TokEof) {
            if (Tok.CurTok == TokFn) {
                Tok.getNextToken();
                if (Tok.CurTok != TokIdent) { std::cerr << "expected fn name\n"; return; }
                std::string FnName = Tok.IdentStr;
                Tok.getNextToken();

                if (Tok.CurTok != TokLParen) { std::cerr << "expected (\n"; return; }
                Tok.getNextToken();

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

                Type* RetTy = Type::getFloatTy(Context);
                if (Tok.CurTok == TokArrow) {
                    Tok.getNextToken();
                    Type* parsed = parseType(Tok);
                    if (parsed) RetTy = parsed;
                }

                if (Tok.CurTok != TokLBrace) { std::cerr << "expected {\n"; return; }

                std::vector<Type*> ArgTypes;
                for (auto &p : Args) ArgTypes.push_back(p.second);
                Function* F = Function::Create(FunctionType::get(RetTy, ArgTypes, false),
                                               Function::ExternalLinkage, FnName, Mod.get());

                BasicBlock* BB = BasicBlock::Create(Context, "entry", F);
                Builder.SetInsertPoint(BB);

                NamedValues.clear();
                NamedAllocas.clear();
                unsigned i = 0;
                for (auto &A : F->args()) {
                    A.setName(Args[i].first);
                    NamedValues[Args[i++].first] = &A;
                }

                Tok.getNextToken(); // consume '{'

                // Parse variable declarations (pure SSA - no allocas unless needed)
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
                        Init = parseVectorLiteral(Tok, cast<FixedVectorType>(VarTy));
                        if (!Init) return;
                    } else if (Tok.CurTok == TokAssign) {
                        Tok.getNextToken();
                        Init = parseExpr(Tok);
                        if (!Init) return;
                    }
                    
                    // Pure SSA: directly bind value, no alloca/store/load
                    if (Init) {
                        NamedValues[VarName] = Init;
                    } else {
                        // No initializer - create undef value
                        NamedValues[VarName] = UndefValue::get(VarTy);
                    }
                    
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

                if (verifyFunction(*F, &errs())) {
                    std::cerr << "Function verification failed\n";
                    return;
                }
            } else {
                Tok.getNextToken();
            }
        }

        Mod->print(errs(), nullptr);

        cantFail(JIT->addIRModule(ThreadSafeModule(std::move(Mod),
            ThreadSafeContext(std::make_unique<LLVMContext>()))));

        auto MainSym = JIT->lookup("main");
        if (!MainSym) {
            std::cerr << "main() not found in JIT module\n";
            return;
        }

        float (*MainPtr)() = (float(*)())MainSym->getValue();
        float result = MainPtr();
        std::cout << "=> " << result << "\n";
    }
};

int main() {
    Compiler C;
    std::cout << "Blaze REPL v3 — vec4/vec8 + swizzling + if/else + for loops + pure SSA!\n";

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