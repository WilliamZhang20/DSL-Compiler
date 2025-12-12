# Blaze DSL

A basic domain-specific language for linear algebra operations.

Still building up from foundational concepts like JIT, REPL, auto-vectorization, inlining, deadcode elimination, memory optimization, etc.

## Getting Started

To compile, run `clang++-18 -std=c++17 program.cpp $(llvm-config-18 --cxxflags --ldflags --libs) -o blaze`

For `simple_JIT.cpp`, run:
- `./blaze`

For `blaze_repl.cpp`:
- Run `./blaze`:
- paste input (or better to also input as file like `./blaze < file.blaze`):
```
fn add(x: i32, y: i32) -> i32 {
    return x + y
}

fn main() -> i32 {
    return add(123, 456) + add(10, 20)
}
```
- After pressing enter to leave a line blank, expect `=> 609` as output.