## Build WASM

```bash
wasm-pack build --target web --out-dir game
cd game
npx serve
```

## Development WASM

```bash
# Terminal 1
cargo watch -- wasm-pack build --target web --dev --out-dir game

# Terminal 2
cd game
npx serve
```

_Note: index.html can be found here: [sotrh#wasm-example](https://sotrh.github.io/learn-wgpu/beginner/tutorial1-window/#wasm-example)_
