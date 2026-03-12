### eval

- server

```bash
# from one process
conda activate <model_env>
python scripts/eval/start_server.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py
```

```bash
# from another process
conda activate <internutopia>
MESA_GL_VERSION_OVERRIDE=4.6 python scripts/eval/eval.py --config scripts/eval/configs/h1_internvla_n1_async_cfg.py

# set config with the following fields
eval_cfg = EvalCfg(
    eval_settings={
        'use_agent_server': True,          # run the model in the same process as the simulator
    },
)
```