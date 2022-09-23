# Usage examples

Windows version can be run by double-click or from cmdline

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

Running with NVIDIA gpu under linux may require explicitely setting `VK_DRIVER_FILES` variable
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
idxs:660178211-926204122 first_elem: 0x21800200 0x43400400 0x87000800 0x0F001001 
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
idxs:160443825-1006631935 first_elem: 0xE11572D0 0xC22AE5E1 0x8455CC43 0x08AB9987 
Runtime error: ERROR_DEVICE_LOST while getting () in context wait_for_fences
```
...hangs in-kernel due to driver


Nvidia Jetson is supported by aarch64 binary
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

# License

memtest_vulkan is licensed similar to `erupt` under the [zlib License](https://github.com/GpuZelenograd/memtest_vulkan/blob/main/LICENSE)
