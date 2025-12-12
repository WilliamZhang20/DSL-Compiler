# Blaze DSL

A basic domain-specific language for linear algebra operations.

## Getting Started

For `simple_JIT.cpp`, run:
- `clang++-18 -std=c++17 main.cpp $(llvm-config-18 --cxxflags --ldflags --libs) -o blaze`
- `./blaze`

For `blaze_repl.cpp`, run:
- same as before, with the input:
```
fn add(x: i32, y: i32) -> i32 {
    return x + y
}

fn main() -> i32 {
    return add(123, 456) + add(10, 20)
}
```
- After pressing enter to leave a line blank, expect `=> 609` as output.