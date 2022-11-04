# [memtest_vulkan](https://github.com/GpuZelenograd/memtest_vulkan/blob/main/Readme.md) - GPU memory testing tool

Opensource tool written in vulkan compute to stress test video memory for stability during overclocking or repair.
Developed as an alternative to OpenCL-based tool [memtestCL](https://github.com/ihaque/memtestCL)

Just start application, wait several minutes and stop testing by Ctrl+C. Detected errors are displayed immediately during test run. Though the image below gives detailed descriptions for errors table - actually it is not needed for 90% of uses. Just run the tool and see if errors are absent or present.

[Prebuilt binaries for windows and linux, including aarch64 NVIDIA Jetson](https://github.com//GpuZelenograd/memtest_vulkan/releases/)

Requires system-provided vulkan loader and vulkan driver supporting Vulkan 1.1 (they are installed with graphics drivers on most OS).
Also requires support of `DEVICE_LOCAL+HOST_COHERENT` memory type from the compute device.

## Usage examples

Windows version can be started by double-click

<a id="usage_screenshot">![WindowsScreenshot](.github/memtest_vulkan_windows_rx580.png)</a>

The example above was stopped by Ctrl+C after showing first error.

Example run from windows command line without errors:
```
C:\Users\galkinvv\Desktop\x86_64-pc-windows-gnu>memtest_vulkan.exe
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x00:00 DevId=0x9A49   8GB Intel(R) Iris(R) Xe Graphics
Testing 1: Bus=0x00:00 DevId=0x9A49   8GB Intel(R) Iris(R) Xe Graphics
      1 iteration. Since last report passed 271.3561ms      written     1.8GB, read:     3.5GB     19.3GB/sec
      5 iteration. Since last report passed 1.0910091s      written     7.0GB, read:    14.0GB     19.2GB/sec
     42 iteration. Since last report passed 10.2049349s     written    64.8GB, read:   129.5GB     19.0GB/sec
    409 iteration. Since last report passed 100.2136744s    written   642.2GB, read:  1284.5GB     19.2GB/sec
    791 iteration. Since last report passed 100.0165577s    written   668.5GB, read:  1337.0GB     20.1GB/sec
   1173 iteration. Since last report passed 100.1249672s    written   668.5GB, read:  1337.0GB     20.0GB/sec
   1551 iteration. Since last report passed 100.0042873s    written   661.5GB, read:  1323.0GB     19.8GB/sec
(Ctrl-C pressed)
memtest_vulkan: no any errors, testing PASSed.
  press any key to continue...
```

Running with NVIDIA gpu under linux may require explicitly setting `VK_DRIVER_FILES` variable
```
[user@host ~]$ VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x01:00 DevId=0x2204   24GB NVIDIA GeForce RTX 3090
Testing 1: Bus=0x01:00 DevId=0x2204   24GB NVIDIA GeForce RTX 3090
      1 iteration. Since last report passed 56.112854ms     written    19.5GB, read:    22.8GB    752.9GB/sec
     19 iteration. Since last report passed 1.011701765s    written   351.0GB, read:   409.5GB    751.7GB/sec
    199 iteration. Since last report passed 10.050222094s   written  3510.0GB, read:  4095.0GB    756.7GB/sec
   1954 iteration. Since last report passed 100.004113065s  written 34222.5GB, read: 39926.2GB    741.5GB/sec
^C
memtest_vulkan: no any errors, testing PASSed.
  press any key to continue...
```

Example run with a single-wire/singe-bit error
```
[user@host ~]$ ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x01:00 DevId=0x1B87   8GB NVIDIA P104-100
Testing 1: Bus=0x01:00 DevId=0x1B87   8GB NVIDIA P104-100
      1 iteration. Since last report passed 52.20479ms      written     3.8GB, read:     7.5GB    215.5GB/sec
     21 iteration. Since last report passed 1.0515038s      written    75.0GB, read:   150.0GB    214.0GB/sec
    216 iteration. Since last report passed 10.021230569s   written   731.2GB, read:  1462.5GB    218.9GB/sec
   2125 iteration. Since last report passed 100.010942973s  written  7158.8GB, read: 14317.5GB    214.7GB/sec
Error found. Mode NEXT_RE_READ, total errors 0x3C7EC3 out of 0x3C000000 (0.39384872%)
Errors address range: 0x9D66148C..=0xDCD3036B  deatils:
         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF
Err1BIdx                 |      1m         |                 |                 
   0x1?                  |      1m         |                 |                 
ErrBiCnt      3m 820k    |                 |                 |                 
MemBiCnt            1   2|  32 249 13645067| 15k 39k  81k142k|219k308k 398k468k
   0x1? 506k502k 448k353k|239k134k  63k 25k|79792113  310  43|   5        1    
actual_ff: 0 actual_max: 0xFFFFFFB7 actual_min: 0x00000730 done_iter_or_err:4294967295 iter:1 calc_param 0x00100107
Error found. Mode INITIAL_READ, total errors 0x7E0C6E out of 0x3C000000 (0.82062860%)
Errors address range: 0x11640B6C4..=0x1DFFFEFFF  deatils:
         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF
Err1BIdx                 |      3m         |                 |                 
   0x1?                  |      3m         |                 |                 
ErrBiCnt      6m   1m    |                 |      51    3 598|  302573   824924
   0x1?  1084402   772471|  22 878    7 152|   1   4    1    |                 
MemBiCnt                7|  43 285 15296317| 19k 50k 107k200k|326k483k 653k817k
   0x1? 949k  1m 999k886k|704k493k 297k149k| 62k 20k 57931263| 185  21    1    
actual_ff: 0 actual_max: 0xFFFFFF46 actual_min: 0x000000B0 done_iter_or_err:4294967295 iter:2160 calc_param 0x8708AB91
Runtime error: ERROR_DEVICE_LOST while getting () in context wait_for_fences
```
...hangs in-kernel due to driver


Jetson is supported by aarch64 binary
```
jetson-nx-alpha :: ~ » ./memtest_vulkan
https://github.com/GpuZelenograd/memtest_vulkan v0.3.0 by GpuZelenograd
To finish testing use Ctrl+C

1: Bus=0x00:00 DevId=0xA5BA03D7   8GB NVIDIA Tegra Xavier (nvgpu)
Testing 1: Bus=0x00:00 DevId=0xA5BA03D7   8GB NVIDIA Tegra Xavier (nvgpu)
      1 iteration. Since last report passed 163.678336ms    written     2.4GB, read:     4.8GB     43.5GB/sec
      7 iteration. Since last report passed 1.045756448s    written    14.2GB, read:    28.5GB     40.9GB/sec
     61 iteration. Since last report passed 10.06722992s    written   128.2GB, read:   256.5GB     38.2GB/sec
    593 iteration. Since last report passed 100.063183744s  written  1263.5GB, read:  2527.0GB     37.9GB/sec
   1121 iteration. Since last report passed 100.043447136s  written  1254.0GB, read:  2508.0GB     37.6GB/sec
^C
memtest_vulkan: no any errors, testing PASSed.
  press any key to continue...
```

## <a id="troubleshooting">Troubleshooting & reporting bugs</a>

The `memtest_vulkan: early exit during init: The library failed to load` message means that your system lacks the Khronos Group Vulkan-Loader library. This library is used as a multiplexer between different drivers provided for different devices and typically is installed during installation of any device-specific vulkan driver. However, some platforms may need explicit installation: for example, to install it on ubuntu 18.04 run `sudo apt install libvulkan1`.
Note that this library itself doesn't depend on any GPU, it is loadable even without any vulkan-capable devices at all. So the error above is a pure software-related error, not related to hardware at all.

If the test fails to start and shows `memtest_vulkan: INIT OR FIRST testing failed due to runtime error` for a known to be vulkan-compatible GPU - there is some incompatibility in vulkan installation. Sometimes this is caused by conflicts between several vulkan drivers installed. Since version v1.3.207 [Khronos vulkan loader supports selecting exact driver with `VK_DRIVER_FILES` environment variable](https://github.com/KhronosGroup/Vulkan-Loader/blob/v1.3.233/docs/LoaderInterfaceArchitecture.md#table-of-debug-environment-variables).

For example on Linux with loader `libvulkan.so` version v1.3.207 or above the test can be run with a specific ICD the following way: `VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json ./memtest_vulkan`.
Also try running with root/admin privileges - this is sometimes required on headless devices.

If neither method helps - enable verbose mode by renaming the executable to `memtest_vulkan_verbose` and running again. The test will output diagnostic information to stdout - please copy it to a new issue at https://github.com/GpuZelenograd/memtest_vulkan/issues.

## <a id="development">New feature development</a>

New ideas in the form of [suggestions via creating issues](https://github.com/GpuZelenograd/memtest_vulkan/issues) or pull requests are welcome. However, note that the tool is designed to be cross-platform, so an optimal way to add temperature and hardware monitoring is [relying on VK_KHR_performance_query extension](https://registry.khronos.org/vulkan/specs/1.3-extensions/man/html/VkPerformanceCounterUnitKHR.html), but unfortunately this extension isn't widely supported by 2022.

If you want to experiment with code modifications, there are two ways to do this:
 - the classic way of cloning repo, editing code and use locally-installed `cargo` to build binaries from modified rust source code
 - or the "fast way to make a small change relying on the github infrastructure". This can be useful if you are not familiar with building rust and don't plan to install it locally
      - fork this repository
      - enable workflows on the actions tab of the forked repository
      - edit&commit code changes (small changes are possible even via editing with a browser)
      - and github will build the binary from your changes for you as the artifacts on the actions tab!

## License

memtest_vulkan is licensed similar to `erupt` under the [zlib License](https://github.com/GpuZelenograd/memtest_vulkan/blob/main/LICENSE)

## Video card memory error kinds (theory)

* The single-bit errors like in an [image above](#usage_screenshot). Such errors are counted in ToggleCnt column 0x01 and the exact bit indices are counted in SingleIdx column. Such errors may be detected by EDC in theory if they occur during transmitting by EDC-enabled part of GPU<->memory wire. But I'm not sure if EDC helps if they occure when transmitting between gpu cache and gpu core or something like this.
* The errors on data-inversion bit (if not detected by EDC). Those should be counted in ToggleCnt columns 0x07/0x08 without SingleIdx info for them.
* The multi-bit transmission errors. Those should be counted in ToggleCnt columns above 0x01, without SingleIdx info for them.
* The errors flipped in the memory chips itself during data storage/"refresh cycles". This may be caused by too big period of refresh or other problems. memtest_vulkan uses a part of memory in a "write once at start but reread every time" pattern - it is the reason fot read GB is more then written GB. If a data flips inside this part of memory - there would be infinite log of error messages marked with "Mode NEXT_RE_READ" (in opposite to Mode INITIAL_READ). Lowering the clocks without restarting test doesn't get rid of such errors.
* The errors on the address-transmission bus. The metest_vulkan is designed to perform reads to the non-sequential series of medium-sized sequential blocks. And if the address is wrongly interpreted by a memory chip - the result is completely garbage from wrong cell. Data-bus EDC can't help here. Those errors typically gives completely random error patterns with normal distribution of bits count and flipped bits (so typical number of flipped bits are 12-20 of 32 and getting 1 flipped bit for this case is extremely unrealistic). The result looks like
```
Error found. Mode INITIAL_READ, total errors 0x2B788 out of 0x18000000 (0.04422069%)
Errors address range: 0x6000E900..=0xBFDFF9FF  iteration:38
values range: 0xFFFFA1A4..=0x0000166F   FFFFFFFF-like count:0    bit-level stats table:
         0x0 0x1  0x2 0x3| 0x4 0x5  0x6 0x7| 0x8 0x9  0xA 0xB| 0xC 0xD  0xE 0xF
SinglIdx                 |                 |                 |                 
TogglCnt                2|   7  18   95 264| 8451786 40056770| 11k 15k  20k 23k
   0x1?  23k 21k  17k 12k|81944859 24701266| 486 248   62  29|   4   2         
1sInValu                3|  19  66  223 700|17683704 6856 11k| 16k 21k  25k 26k
   0x1?  23k 17k  12k6327|2883 917  282  64|   9             |
```
* Other critical errors inside memory chips or memory controller. This gives normal distributions for TogglCnt, but for 1sInValu the distribution may be different - since critical internal errors may be reported by some fixed patterns (0x00000000, 0xFFFFFFFF - for some EDC problems, 0x0BADAC?? - for some NVIDIA problems).
* Memory errors in the areas where error counts are stored)) This often shows as millions of errors in all table entries, typically with the total errors greater than tested memory size. Such results are numerically garbage, but means that the gpu/memory is really mostly non-functional.
* The errors in GPU during calculation of addresses and desired values or in value comparison. This can lead to any pattern of reporting at all, since the logic of a program is broken.
