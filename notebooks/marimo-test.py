# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.0",
#     "pyzmq>=27.1.0",
# ]
# ///

import marimo

__generated_with = "0.19.10"
app = marimo.App()


@app.cell
def _():
    print("hello world")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
