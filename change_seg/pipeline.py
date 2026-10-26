import csv
import io
import json
import os
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .data import write_json
from .engine import emit, package, summarize_cv


def assert_gpu_free(gpu):
    result = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"],
                            check=True, capture_output=True, text=True)
    gpu_uuid, used = next(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"],
                               check=True, capture_output=True, text=True)
    active = list(csv.reader(io.StringIO(processes.stdout), skipinitialspace=True))
    if int(used) > 128 or any(row and row[0] == gpu_uuid for row in active):
        raise RuntimeError(f"GPU {gpu} 已被占用，停止启动新任务: {used} MiB")


def run_pipeline(config, config_path, gpus):
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("需要不重复的空闲 GPU 编号")
    root = Path(config["run_root"])
    if (root / "pipeline_status.json").exists():
        raise FileExistsError("pipeline 已有运行记录，请检查已有进程与 checkpoint，不自动覆盖")
    for gpu in gpus:
        assert_gpu_free(gpu)
    write_json(root / "run_config.json", config)
    sources = {str(p): p.read_text() for p in Path("change_seg").glob("*.py")}
    write_json(root / "source_snapshot.json", sources)
    state = {"pid": os.getpid(), "gpus": gpus, "status": "cross_validation", "jobs": {}}
    pending = [(fold, None) for fold in range(config["folds"])]
    phase = "cv"
    active = {}
    available = list(gpus)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        try:
            while pending or active:
                while pending and available:
                    fold, epochs = pending.pop(0)
                    gpu = available.pop(0)
                    assert_gpu_free(gpu)
                    name = "full" if fold == -1 else f"fold_{fold}"
                    log = (root / f"{name}.log").open("a", buffering=1)
                    environment = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="2",
                                       OPENBLAS_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
                    command = [sys.executable, "-u", "-m", "change_seg", "--config", str(config_path),
                               "train", "--fold", str(fold)]
                    if epochs is not None:
                        command += ["--epochs", str(epochs)]
                    process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT)
                    future = pool.submit(process.wait)
                    active[future] = (process, gpu, name, log)
                    state["jobs"][name] = {"pid": process.pid, "gpu": gpu, "status": "running", "log": str(root / f"{name}.log")}
                    write_json(root / "pipeline_status.json", state)
                    emit("job_started", job=name, pid=process.pid, gpu=gpu)
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    process, gpu, name, log = active.pop(future)
                    code = future.result()
                    log.close()
                    state["jobs"][name].update(status="completed" if code == 0 else "failed", exit_code=code)
                    write_json(root / "pipeline_status.json", state)
                    if code:
                        raise RuntimeError(f"{name} 失败，退出码 {code}，检查 {root / (name + '.log')}")
                    available.append(gpu)
                    emit("job_completed", job=name, gpu=gpu)
                if not pending and not active and phase == "cv":
                    summary = summarize_cv(config)
                    pending = [(-1, summary["full_epochs"])]
                    state["status"] = "full_training"
                    phase = "full"
                    write_json(root / "pipeline_status.json", state)
            state["package"] = package(config)
            state["status"] = "completed"
            write_json(root / "pipeline_status.json", state)
            emit("pipeline_complete", **state["package"])
        except BaseException as error:
            state["status"] = "failed"
            state["error"] = repr(error)
            for process, gpu, name, log in active.values():
                if process.poll() is None:
                    process.terminate()
                    process.wait()
                log.close()
                state["jobs"][name].update(status="terminated", exit_code=process.returncode)
            write_json(root / "pipeline_status.json", state)
            raise
