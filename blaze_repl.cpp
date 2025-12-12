// Tiny parser + REPL for Blaze
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <iostream>
#include <sstream>
#include <map>
#include <cctype>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

// ──────────────────────────────────────────────────────────────
// Tiny hand-rolled tokenizer
// ──────────────────────────────────────────────────────────────
enum Token { TokEof, TokFn, TokIdent, TokNumber, TokLParen, TokRParen,
             TokLBrace, TokRBrace, TokReturn, TokColon, TokArrow, TokComma, TokSemi, TokPlus };

struct Tokenizer {
    const char* Ptr;
    std::string IdentStr;
    int NumVal;
    Token CurTok;

    Tokenizer(const std::string& Input) : Ptr(Input.c_str()), CurTok(TokEof) {
        getNextToken(); // Prime the first token
    }

    Token getNextToken() {
        while (*Ptr && std::isspace(*Ptr)) ++Ptr;
        if (!*Ptr) return CurTok = TokEof;

        if (std::isalpha(*Ptr)) {
            IdentStr = *Ptr++;
            while (std::isalnum(*Ptr) || *Ptr == '_') IdentStr += *Ptr++;
            if (IdentStr == "fn") return CurTok = TokFn;
            if (IdentStr == "return") return CurTok = TokReturn;
            return CurTok = TokIdent;
        }

        if (std::isdigit(*Ptr)) {
            NumVal = strtol(Ptr, const_cast<char**>(&Ptr), 10);
            return CurTok = TokNumber;
        }

        char c = *Ptr++;
        switch (c) {
            case '(': return CurTok = TokLParen;
            case ')': return CurTok = TokRParen;
            case '{': return CurTok = TokLBrace;
            case '}': return CurTok = TokRBrace;
            case ':': return CurTok = TokColon;
            case ',': return CurTok = TokComma;
            case ';': return CurTok = TokSemi;
            case '+': return CurTok = TokPlus;
            case '-':
                if (*Ptr == '>') { ++Ptr; return CurTok = TokArrow; }
                std::cerr << "Unknown char: " << c << "\n";
                return CurTok = TokEof;
            default:
                std::cerr << "Unknown char: " << c << "\n";
                return CurTok = TokEof;
        }
    }
};

// ──────────────────────────────────────────────────────────────
// Very small compiler
// ──────────────────────────────────────────────────────────────
class Compiler {
    LLVMContext Context;
    std::unique_ptr<Module> Mod;
    IRBuilder<> Builder;
    std::unique_ptr<LLJIT> JIT;
    std::map<std::string, Value*> NamedValues;

public:
    Compiler() : Builder(Context) {
        InitializeNativeTarget();
        InitializeNativeTargetAsmPrinter();
        Mod = std::make_unique<Module>("blaze_repl", Context);
        JIT = cantFail(LLJITBuilder().create());
    }

    Function* createFunction(const std::string& Name, Type* RetTy,
                             const std::vector<std::pair<std::string, Type*>>& Args) {
        std::vector<Type*> ArgTypes;
        for (auto& A : Args) ArgTypes.push_back(A.second);

        auto* FT = FunctionType::get(RetTy, ArgTypes, false);
        Function* F = Function::Create(FT, Function::ExternalLinkage, Name, Mod.get());

        // Name arguments
        unsigned Idx = 0;
        for (auto& Arg : F->args()) {
            Arg.setName(Args[Idx++].first);
        }
        return F;
    }

    void compile(const std::string& Source) {
        Tokenizer Tok(Source);

        while (Tok.CurTok == TokFn) {
            // fn name ( arg: type, ... ) -> ret_type { ... }
            Tok.getNextToken();
            if (Tok.CurTok != TokIdent) { std::cerr << "expected fn name\n"; return; }
            std::string FnName = Tok.IdentStr;

            Tok.getNextToken();
            if (Tok.CurTok != TokLParen) { std::cerr << "expected (\n"; return; }
            std::vector<std::pair<std::string, Type*>> Args;
            Tok.getNextToken();
            if (Tok.CurTok != TokRParen) {
                do {
                    if (Tok.CurTok != TokIdent) { std::cerr << "expected arg name\n"; return; }
                    std::string ArgName = Tok.IdentStr;
                    Tok.getNextToken();
                    if (Tok.CurTok != TokColon) { std::cerr << "expected :\n"; return; }
                    Tok.getNextToken();
                    if (Tok.CurTok != TokIdent || Tok.IdentStr != "i32") { std::cerr << "only i32 supported\n"; return; }
                    Args.emplace_back(ArgName, Type::getInt32Ty(Context));
                    Tok.getNextToken();
                    if (Tok.CurTok == TokComma) Tok.getNextToken();
                } while (Tok.CurTok != TokRParen);
            }

            Tok.getNextToken(); // -> or {
            Type* RetTy = Type::getInt32Ty(Context);
            if (Tok.CurTok == TokArrow) {
                Tok.getNextToken();
                if (Tok.CurTok != TokIdent || Tok.IdentStr != "i32") { std::cerr << "only i32 return\n"; return; }
                Tok.getNextToken();
            }

            if (Tok.CurTok != TokLBrace) { std::cerr << "expected {\n"; return; }

            Function* F = createFunction(FnName, RetTy, Args);

            BasicBlock* BB = BasicBlock::Create(Context, "entry", F);
            Builder.SetInsertPoint(BB);
            NamedValues.clear();
            for (auto& Arg : F->args()) NamedValues[std::string(Arg.getName())] = &Arg;

            // body — only simple return expr for now
            Tok.getNextToken();
            if (Tok.CurTok == TokReturn) {
                Tok.getNextToken();
                Value* V = parseExpr(Tok);
                if (!V) { std::cerr << "failed to parse expression\n"; return; }
                Builder.CreateRet(V);
            } else {
                std::cerr << "only return expr supported so far\n"; return;
            }

            if (Tok.CurTok == TokSemi) Tok.getNextToken();
            if (Tok.CurTok != TokRBrace) { std::cerr << "expected }\n"; return; }
            verifyFunction(*F);
            Tok.getNextToken(); // next fn or eof
        }

        // Add module to JIT
        auto TSCtx = std::make_unique<LLVMContext>();
        cantFail(JIT->addIRModule(ThreadSafeModule(std::move(Mod),
            ThreadSafeContext(std::move(TSCtx)))));

        // If there's a main, run it
        auto MainSym = JIT->lookup("main");
        if (MainSym) {
            int (*MainPtr)() = (int(*)())MainSym->getValue();
            std::cout << "=> " << MainPtr() << "\n";
        }
        
        // Reset for next compilation
        Mod = std::make_unique<Module>("blaze_repl", Context);
    }

private:
    // Parse primary expression (number, variable, or function call)
    Value* parsePrimary(Tokenizer& Tok) {
        if (Tok.CurTok == TokNumber) {
            Value* V = ConstantInt::get(Context, APInt(32, Tok.NumVal));
            Tok.getNextToken();
            return V;
        }
        
        if (Tok.CurTok == TokIdent) {
            std::string Name = Tok.IdentStr;
            Tok.getNextToken();
            
            // Check if it's a function call
            if (Tok.CurTok == TokLParen) {
                Tok.getNextToken(); // consume (
                
                // Get the function
                Function* CalleeF = Mod->getFunction(Name);
                if (!CalleeF) {
                    std::cerr << "Unknown function: " << Name << "\n";
                    return nullptr;
                }
                
                // Parse arguments
                std::vector<Value*> ArgsV;
                if (Tok.CurTok != TokRParen) {
                    while (true) {
                        Value* Arg = parseExpr(Tok);
                        if (!Arg) return nullptr;
                        ArgsV.push_back(Arg);
                        
                        if (Tok.CurTok == TokRParen) break;
                        if (Tok.CurTok != TokComma) {
                            std::cerr << "Expected ')' or ',' in argument list\n";
                            return nullptr;
                        }
                        Tok.getNextToken(); // consume comma
                    }
                }
                Tok.getNextToken(); // consume )
                
                return Builder.CreateCall(CalleeF, ArgsV, "calltmp");
            }
            
            // It's a variable
            if (NamedValues.find(Name) == NamedValues.end()) {
                std::cerr << "Unknown variable: " << Name << "\n";
                return nullptr;
            }
            return NamedValues[Name];
        }
        
        std::cerr << "Expected number, variable, or function call\n";
        return nullptr;
    }
    
    // Parse addition expression
    Value* parseExpr(Tokenizer& Tok) {
        Value* LHS = parsePrimary(Tok);
        if (!LHS) return nullptr;
        
        // Handle addition
        while (Tok.CurTok == TokPlus) {
            Tok.getNextToken();
            Value* RHS = parsePrimary(Tok);
            if (!RHS) return nullptr;
            LHS = Builder.CreateAdd(LHS, RHS, "addtmp");
        }
        
        return LHS;
    }
};

// ──────────────────────────────────────────────────────────────
// REPL
// ──────────────────────────────────────────────────────────────
int main() {
    Compiler C;
    std::cout << "Blaze REPL — type Blaze code, empty line to run\n";

    std::string Line, Source;
    while (std::cout << "> ", std::getline(std::cin, Line)) {
        if (Line.empty()) {
            if (!Source.empty()) {
                C.compile(Source);
                Source.clear();
            }
        } else {
            Source += Line + "\n";
        }
    }
}