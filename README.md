# fertool

To install dependencies:

```bash
bun install
```

To run:

```bash
bun run index.ts
```

This project was created using `bun init` in bun v1.2.5. [Bun](https://bun.sh) is a fast all-in-one JavaScript runtime.

## Run executable

```bash
cd dist

./fertool-darwin-arm64 -i ../.test/pictures2 -o ../.test/results_2.csv -m ../model/yolo.onnx --debug

```
