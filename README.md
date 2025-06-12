Orignial author:　https://github.com/WanyuGroup/TIFS2022-Solitude

paper: https://wanyu-lin.github.io/assets/pdf/wanyu-tifs2022.pdf

our presetaion ppt: https://www.canva.com/design/DAGqDxqvNDw/1Nqfp_GlSIMkaVxImjhUNA/view?utm_content=DAGqDxqvNDw&utm_campaign=designshare&utm_medium=link2&utm_source=uniquelinks&utlId=h796dc1f22f

---

### Our Running Set up

```
python main.py -d lastfm --model gcn -ex 1.0 -ee 7.9  --orphic True
```

---

### Using Version

torch               2.0.1+cu118

torch-geometric     2.6.1

pandas              2.3.0

numpy               1.26.4

seaborn             0.13.2

---

### Modification

We add a GPU time counter and record the GPU memory usage at both.

We chamge the second one from using dense matrix to sparce mathrix to improve the efficiency.

 
