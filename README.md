# Task-Equivariant Graph Few-shot Learning (TEG)

Code for experiments based on **Task-Equivariant Graph Few-shot Learning (TEG)**.

## Environment

The Conda environment used for this project is provided in `environment.yml`.

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate python310
```

## How to Run

Make the shell script executable:

```bash
chmod +x shell.sh
```

Run the experiment:

```bash
./shell.sh
```

The current configuration runs:

```bash
python -u main.py --dataset corafull --way 5 --shot 3 > "${LOG_FILE}" 2>&1 &
```

## Logs

The experiment runs in the background and automatically saves logs in:

```text
out/YYMMDD/HHMMSSmmm.log
```

For example:

```text
out/261001/221530123.log
```

When `shell.sh` is executed, the log path and process ID are printed to the terminal.