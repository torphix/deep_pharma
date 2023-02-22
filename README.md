## About
A small project that implements residual gated graph convolutional networks https://arxiv.org/pdf/1711.07553v2.pdf in pytorch and predicts ADMET properties of molecules
Comes with a UI built in [gradio](https://gradio.app/docs/)

Uses a multi-headed network approach as using a singular backbone allows for feature sharing and possibly better results.


## Commands
```python main.py train ```
```python main.py evaulate```

### UI 
```python ui.py ```

## References
```
TDC Dataset: https://github.com/mims-harvard/TDC
@article{Huang2021tdc,
  title={Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development},
  author={Huang, Kexin and Fu, Tianfan and Gao, Wenhao and Zhao, Yue and Roohani, Yusuf and Leskovec, Jure and Coley, 
          Connor W and Xiao, Cao and Sun, Jimeng and Zitnik, Marinka},
  journal={Proceedings of Neural Information Processing Systems, NeurIPS Datasets and Benchmarks},
  year={2021}
}
```
Implicit valence = atom valence - valence from bond connections
