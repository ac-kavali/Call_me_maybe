## Execution format:
```sh
python3 main.py  -f /file.json -i prompt.json -o output.json
```

## uv uses 

1. **create the .venv in the goinfre**
    ```shell
    uv venv ~/goinfre/venv
    ```
2. **then link the project .venv**

    ```sh
    ln -s ~/goinfre/venv venv
   ```
   
export HF_HOME=~/goinfre/hf_cache
export HUGGINGFACE_HUB_CACHE=~/goinfre/hf_cache