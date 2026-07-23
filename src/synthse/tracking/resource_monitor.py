from __future__ import annotations

import os
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch

try:
    import pynvml
except ImportError:
    pynvml = None


MEBIBYTE = 1024**2


@dataclass
class ResourceSample:
    """
    One resource-monitoring sample collected during an operation.
    """

    sample_index: int
    elapsed_seconds: float

    gpu_power_w: float | None
    gpu_utilization_percent: float | None
    gpu_memory_used_mb: float | None
    gpu_temperature_c: float | None

    process_cpu_percent: float
    process_rss_mb: float
    system_ram_used_mb: float


def round_optional(
    value: float | None,
    digits: int = 6,
) -> float | None:
    """
    Round a value when it is available.
    """
    if value is None:
        return None

    return round(
        float(value),
        digits,
    )


def mean_optional(
    values: list[float | None],
) -> float | None:
    """
    Calculate the arithmetic mean while ignoring unavailable values.
    """
    available_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not available_values:
        return None

    return statistics.fmean(
        available_values
    )


def max_optional(
    values: list[float | None],
) -> float | None:
    """
    Calculate the maximum while ignoring unavailable values.
    """
    available_values = [
        float(value)
        for value in values
        if value is not None
    ]

    if not available_values:
        return None

    return max(
        available_values
    )


class ResourceMonitor:
    """
    Monitor GPU, CPU, RAM and VRAM usage during one operation.

    The monitor collects periodic samples in a background thread and
    returns aggregate metrics when stopped.

    GPU metrics are collected through NVML when available.

    CPU and RAM metrics are collected through psutil.

    PyTorch CUDA peak-memory metrics are collected through torch.cuda.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        sample_interval_seconds: float = 0.2,
        process_id: int | None = None,
        enabled: bool = True,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ValueError(
                "sample_interval_seconds must be "
                "greater than zero."
            )

        self.gpu_index = int(
            gpu_index
        )

        self.sample_interval_seconds = float(
            sample_interval_seconds
        )

        self.process_id = (
            int(process_id)
            if process_id is not None
            else os.getpid()
        )

        self.enabled = bool(
            enabled
        )

        self.process = psutil.Process(
            self.process_id
        )

        self.samples: list[
            ResourceSample
        ] = []

        self.phase = ""

        self._running = False

        self._thread: threading.Thread | None = None

        self._stop_event = threading.Event()

        self._start_time = 0.0

        self._process_cpu_times_start: Any = None

        self._process_rss_start_mb: float | None = None

        self._system_ram_used_start_mb: float | None = None

        self._gpu_temperature_start_c: float | None = None

        self._gpu_power_limit_w: float | None = None

        self._gpu_energy_start_mj: float | None = None

        self._gpu_energy_counter_supported = False

        self._nvml_initialized = False

        self._gpu_handle: Any = None

    def __enter__(
        self,
    ) -> ResourceMonitor:
        self.start()

        return self

    def __exit__(
        self,
        exception_type: Any,
        exception_value: Any,
        traceback: Any,
    ) -> None:
        self.stop()

    def _initialize_nvml(
        self,
    ) -> None:
        """
        Initialize NVML and resolve the selected GPU.

        Missing NVML support does not prevent CPU and RAM monitoring.
        """
        if not self.enabled:
            return

        if pynvml is None:
            print(
                "ResourceMonitor: nvidia-ml-py is not installed. "
                "GPU metrics will be unavailable."
            )
            return

        try:
            pynvml.nvmlInit()

            self._gpu_handle = (
                pynvml.nvmlDeviceGetHandleByIndex(
                    self.gpu_index
                )
            )

            self._nvml_initialized = True

        except Exception as error:
            self._gpu_handle = None
            self._nvml_initialized = False

            print(
                "ResourceMonitor: NVML initialization failed. "
                f"GPU metrics will be unavailable: {error}"
            )

    def _shutdown_nvml(
        self,
    ) -> None:
        """
        Shut down NVML when it was initialized by this monitor.
        """
        if (
            pynvml is None
            or not self._nvml_initialized
        ):
            return

        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

        self._nvml_initialized = False
        self._gpu_handle = None

    def _read_gpu_power_w(
        self,
    ) -> float | None:
        """
        Return the current GPU power draw in watts.
        """
        if self._gpu_handle is None:
            return None

        try:
            power_mw = (
                pynvml.nvmlDeviceGetPowerUsage(
                    self._gpu_handle
                )
            )

            return float(
                power_mw
            ) / 1000.0

        except Exception:
            return None

    def _read_gpu_utilization_percent(
        self,
    ) -> float | None:
        """
        Return the current GPU-compute utilization percentage.
        """
        if self._gpu_handle is None:
            return None

        try:
            utilization = (
                pynvml.nvmlDeviceGetUtilizationRates(
                    self._gpu_handle
                )
            )

            return float(
                utilization.gpu
            )

        except Exception:
            return None

    def _read_gpu_memory_used_mb(
        self,
    ) -> float | None:
        """
        Return total GPU memory used according to NVML.
        """
        if self._gpu_handle is None:
            return None

        try:
            memory_info = (
                pynvml.nvmlDeviceGetMemoryInfo(
                    self._gpu_handle
                )
            )

            return float(
                memory_info.used
            ) / MEBIBYTE

        except Exception:
            return None

    def _read_gpu_temperature_c(
        self,
    ) -> float | None:
        """
        Return the current GPU temperature in Celsius.
        """
        if self._gpu_handle is None:
            return None

        try:
            temperature = (
                pynvml.nvmlDeviceGetTemperature(
                    self._gpu_handle,
                    pynvml.NVML_TEMPERATURE_GPU,
                )
            )

            return float(
                temperature
            )

        except Exception:
            return None

    def _read_gpu_power_limit_w(
        self,
    ) -> float | None:
        """
        Return the enforced or configured GPU power limit in watts.
        """
        if self._gpu_handle is None:
            return None

        try:
            power_limit_mw = (
                pynvml.nvmlDeviceGetEnforcedPowerLimit(
                    self._gpu_handle
                )
            )

            return float(
                power_limit_mw
            ) / 1000.0

        except Exception:
            try:
                power_limit_mw = (
                    pynvml.nvmlDeviceGetPowerManagementLimit(
                        self._gpu_handle
                    )
                )

                return float(
                    power_limit_mw
                ) / 1000.0

            except Exception:
                return None

    def _read_gpu_total_energy_mj(
        self,
    ) -> float | None:
        """
        Return the cumulative GPU energy counter in millijoules.

        The function is not supported by every NVIDIA GPU or driver.
        """
        if self._gpu_handle is None:
            return None

        try:
            total_energy_mj = (
                pynvml.nvmlDeviceGetTotalEnergyConsumption(
                    self._gpu_handle
                )
            )

            return float(
                total_energy_mj
            )

        except Exception:
            return None

    def _read_process_rss_mb(
        self,
    ) -> float:
        """
        Return the process resident-set size in MiB.
        """
        return float(
            self.process.memory_info().rss
        ) / MEBIBYTE

    @staticmethod
    def _read_system_ram_used_mb(
    ) -> float:
        """
        Return total system RAM currently used in MiB.
        """
        return float(
            psutil.virtual_memory().used
        ) / MEBIBYTE

    def _reset_torch_cuda_peaks(
        self,
    ) -> None:
        """
        Reset PyTorch CUDA peak-memory counters.
        """
        if not torch.cuda.is_available():
            return

        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    @staticmethod
    def _read_torch_cuda_peaks(
    ) -> tuple[
        float | None,
        float | None,
    ]:
        """
        Return PyTorch peak allocated and reserved VRAM in MiB.
        """
        if not torch.cuda.is_available():
            return None, None

        try:
            torch.cuda.synchronize()

            allocated_mb = (
                torch.cuda.max_memory_allocated()
                / MEBIBYTE
            )

            reserved_mb = (
                torch.cuda.max_memory_reserved()
                / MEBIBYTE
            )

            return (
                float(allocated_mb),
                float(reserved_mb),
            )

        except Exception:
            return None, None

    def _collect_sample(
        self,
        sample_index: int,
    ) -> ResourceSample:
        """
        Collect one resource sample.
        """
        elapsed_seconds = (
            time.perf_counter()
            - self._start_time
        )

        return ResourceSample(
            sample_index=sample_index,
            elapsed_seconds=elapsed_seconds,
            gpu_power_w=self._read_gpu_power_w(),
            gpu_utilization_percent=(
                self._read_gpu_utilization_percent()
            ),
            gpu_memory_used_mb=(
                self._read_gpu_memory_used_mb()
            ),
            gpu_temperature_c=(
                self._read_gpu_temperature_c()
            ),
            process_cpu_percent=float(
                self.process.cpu_percent(
                    interval=None
                )
            ),
            process_rss_mb=(
                self._read_process_rss_mb()
            ),
            system_ram_used_mb=(
                self._read_system_ram_used_mb()
            ),
        )

    def _sampling_loop(
        self,
    ) -> None:
        """
        Collect samples until the monitor is stopped.
        """
        sample_index = 0

        while not self._stop_event.is_set():
            try:
                sample = self._collect_sample(
                    sample_index
                )

                self.samples.append(
                    sample
                )

                sample_index += 1

            except (
                psutil.NoSuchProcess,
                psutil.AccessDenied,
            ):
                break

            self._stop_event.wait(
                self.sample_interval_seconds
            )

    def start(
        self,
        phase: str = "generation",
    ) -> None:
        """
        Start monitoring one operation.
        """
        if not self.enabled:
            self.phase = phase
            self._start_time = (
                time.perf_counter()
            )
            return

        if self._running:
            raise RuntimeError(
                "ResourceMonitor is already running."
            )

        self.phase = str(
            phase
        )

        self.samples = []

        self._stop_event.clear()

        self._initialize_nvml()

        self._reset_torch_cuda_peaks()

        self._start_time = (
            time.perf_counter()
        )

        self._process_cpu_times_start = (
            self.process.cpu_times()
        )

        self._process_rss_start_mb = (
            self._read_process_rss_mb()
        )

        self._system_ram_used_start_mb = (
            self._read_system_ram_used_mb()
        )

        self._gpu_temperature_start_c = (
            self._read_gpu_temperature_c()
        )

        self._gpu_power_limit_w = (
            self._read_gpu_power_limit_w()
        )

        self._gpu_energy_start_mj = (
            self._read_gpu_total_energy_mj()
        )

        self._gpu_energy_counter_supported = (
            self._gpu_energy_start_mj
            is not None
        )

        # Initialize psutil's non-blocking CPU percentage counter.
        self.process.cpu_percent(
            interval=None
        )

        self._running = True

        self._thread = threading.Thread(
            target=self._sampling_loop,
            name=(
                f"resource-monitor-{self.phase}"
            ),
            daemon=True,
        )

        self._thread.start()

    def _integrate_power_samples_j(
        self,
    ) -> float | None:
        """
        Integrate sampled GPU power using the trapezoidal rule.
        """
        valid_samples = [
            sample
            for sample in self.samples
            if sample.gpu_power_w is not None
        ]

        if len(valid_samples) < 2:
            return None

        energy_j = 0.0

        for previous, current in zip(
            valid_samples,
            valid_samples[1:],
            strict=False,
        ):
            delta_seconds = (
                current.elapsed_seconds
                - previous.elapsed_seconds
            )

            if delta_seconds <= 0:
                continue

            mean_power_w = (
                float(previous.gpu_power_w)
                + float(current.gpu_power_w)
            ) / 2.0

            energy_j += (
                mean_power_w
                * delta_seconds
            )

        return energy_j

    def stop(
        self,
    ) -> dict[str, Any]:
        """
        Stop monitoring and return aggregate metrics.
        """
        if not self.enabled:
            elapsed_seconds = (
                time.perf_counter()
                - self._start_time
            )

            return {
                "monitoring_enabled": False,
                "monitoring_phase": self.phase,
                "monitoring_duration_seconds": (
                    round_optional(
                        elapsed_seconds
                    )
                ),
            }

        if not self._running:
            raise RuntimeError(
                "ResourceMonitor is not running."
            )

        self._stop_event.set()

        if self._thread is not None:
            self._thread.join(
                timeout=(
                    self.sample_interval_seconds
                    * 3
                )
            )

        # Collect one final sample so the end state is represented.
        try:
            final_sample = self._collect_sample(
                len(self.samples)
            )

            self.samples.append(
                final_sample
            )
        except Exception:
            pass

        end_time = time.perf_counter()

        monitoring_duration_seconds = (
            end_time
            - self._start_time
        )

        process_cpu_times_end = (
            self.process.cpu_times()
        )

        process_rss_end_mb = (
            self._read_process_rss_mb()
        )

        system_ram_used_end_mb = (
            self._read_system_ram_used_mb()
        )

        gpu_temperature_end_c = (
            self._read_gpu_temperature_c()
        )

        gpu_energy_end_mj = (
            self._read_gpu_total_energy_mj()
        )

        (
            torch_peak_allocated_vram_mb,
            torch_peak_reserved_vram_mb,
        ) = self._read_torch_cuda_peaks()

        gpu_energy_j: float | None = None

        energy_measurement_method = (
            "unavailable"
        )

        if (
            self._gpu_energy_counter_supported
            and self._gpu_energy_start_mj
            is not None
            and gpu_energy_end_mj
            is not None
            and gpu_energy_end_mj
            >= self._gpu_energy_start_mj
        ):
            gpu_energy_j = (
                gpu_energy_end_mj
                - self._gpu_energy_start_mj
            ) / 1000.0

            energy_measurement_method = (
                "nvml_total_energy_counter"
            )

        else:
            gpu_energy_j = (
                self._integrate_power_samples_j()
            )

            if gpu_energy_j is not None:
                energy_measurement_method = (
                    "integrated_power_samples"
                )

        gpu_power_samples = [
            sample.gpu_power_w
            for sample in self.samples
        ]

        gpu_utilization_samples = [
            sample.gpu_utilization_percent
            for sample in self.samples
        ]

        gpu_memory_samples = [
            sample.gpu_memory_used_mb
            for sample in self.samples
        ]

        gpu_temperature_samples = [
            sample.gpu_temperature_c
            for sample in self.samples
        ]

        process_cpu_samples = [
            sample.process_cpu_percent
            for sample in self.samples
        ]

        process_rss_samples = [
            sample.process_rss_mb
            for sample in self.samples
        ]

        system_ram_samples = [
            sample.system_ram_used_mb
            for sample in self.samples
        ]

        process_cpu_user_seconds = (
            process_cpu_times_end.user
            - self._process_cpu_times_start.user
        )

        process_cpu_system_seconds = (
            process_cpu_times_end.system
            - self._process_cpu_times_start.system
        )

        process_rss_peak_values = [
            value
            for value in process_rss_samples
        ]

        if (
            self._process_rss_start_mb
            is not None
        ):
            process_rss_peak_values.append(
                self._process_rss_start_mb
            )

        process_rss_peak_values.append(
            process_rss_end_mb
        )

        system_ram_peak_values = [
            value
            for value in system_ram_samples
        ]

        if (
            self._system_ram_used_start_mb
            is not None
        ):
            system_ram_peak_values.append(
                self._system_ram_used_start_mb
            )

        system_ram_peak_values.append(
            system_ram_used_end_mb
        )

        result = {
            "monitoring_enabled": True,
            "monitoring_phase": self.phase,
            "monitoring_duration_seconds": (
                round_optional(
                    monitoring_duration_seconds
                )
            ),
            "monitoring_sample_count": len(
                self.samples
            ),
            "gpu_energy_j": round_optional(
                gpu_energy_j
            ),
            "gpu_energy_wh": round_optional(
                (
                    gpu_energy_j / 3600.0
                    if gpu_energy_j is not None
                    else None
                ),
                digits=9,
            ),
            "gpu_energy_measurement_method": (
                energy_measurement_method
            ),
            "gpu_power_mean_w": round_optional(
                mean_optional(
                    gpu_power_samples
                )
            ),
            "gpu_power_max_w": round_optional(
                max_optional(
                    gpu_power_samples
                )
            ),
            "gpu_utilization_mean_percent": (
                round_optional(
                    mean_optional(
                        gpu_utilization_samples
                    )
                )
            ),
            "gpu_utilization_max_percent": (
                round_optional(
                    max_optional(
                        gpu_utilization_samples
                    )
                )
            ),
            "gpu_temperature_start_c": (
                round_optional(
                    self._gpu_temperature_start_c
                )
            ),
            "gpu_temperature_max_c": (
                round_optional(
                    max_optional(
                        gpu_temperature_samples
                    )
                )
            ),
            "gpu_temperature_end_c": (
                round_optional(
                    gpu_temperature_end_c
                )
            ),
            "gpu_power_limit_w": (
                round_optional(
                    self._gpu_power_limit_w
                )
            ),
            "torch_peak_allocated_vram_mb": (
                round_optional(
                    torch_peak_allocated_vram_mb
                )
            ),
            "torch_peak_reserved_vram_mb": (
                round_optional(
                    torch_peak_reserved_vram_mb
                )
            ),
            "nvml_peak_used_vram_mb": (
                round_optional(
                    max_optional(
                        gpu_memory_samples
                    )
                )
            ),
            "process_cpu_mean_percent": (
                round_optional(
                    mean_optional(
                        [
                            float(value)
                            for value
                            in process_cpu_samples
                        ]
                    )
                )
            ),
            "process_cpu_max_percent": (
                round_optional(
                    max_optional(
                        [
                            float(value)
                            for value
                            in process_cpu_samples
                        ]
                    )
                )
            ),
            "process_cpu_user_seconds": (
                round_optional(
                    process_cpu_user_seconds
                )
            ),
            "process_cpu_system_seconds": (
                round_optional(
                    process_cpu_system_seconds
                )
            ),
            "process_rss_start_mb": (
                round_optional(
                    self._process_rss_start_mb
                )
            ),
            "process_rss_peak_mb": (
                round_optional(
                    max(
                        process_rss_peak_values
                    )
                )
            ),
            "process_rss_end_mb": (
                round_optional(
                    process_rss_end_mb
                )
            ),
            "system_ram_used_start_mb": (
                round_optional(
                    self._system_ram_used_start_mb
                )
            ),
            "system_ram_used_peak_mb": (
                round_optional(
                    max(
                        system_ram_peak_values
                    )
                )
            ),
            "system_ram_used_end_mb": (
                round_optional(
                    system_ram_used_end_mb
                )
            ),
        }

        self._running = False

        self._shutdown_nvml()

        return result

    def get_raw_samples(
        self,
    ) -> list[dict[str, Any]]:
        """
        Return raw samples as serializable dictionaries.
        """
        return [
            {
                "phase": self.phase,
                **asdict(sample),
            }
            for sample in self.samples
        ]