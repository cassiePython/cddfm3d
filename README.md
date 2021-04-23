# Cross-Domain and Disentangled Face Manipulation with 3D Guidance

<img src='imgs/teaser.png'/>

### [Project Page](https://cassiepython.github.io/sigasia/cddfm3d.html) | [Paper (ArXiv)](https://arxiv.org/abs/2104.11228)


[Can Wang](https://cassiepython.github.io/)<sup>1</sup>,
[Menglei Chai](https://mlchai.com/)<sup>2</sup>,
[Mingming He](http://mingminghe.com/)<sup>3</sup>,
[Dongdong Chen](http://www.dongdongchen.bid/)<sup>4</sup>,
[Jing Liao](https://liaojing.github.io/html/)<sup>1</sup> <br>
<sup>1</sup>City University of Hong Kong, <sup>2</sup>Creative Vision, Snap Inc., <sup>3</sup>USC Institute for Creative Technologies, <sup>4</sup>Microsoft Cloud AI


## Abstract
<img src='imgs/Pipeline.png'/>
Face image manipulation via three-dimensional guidance has been widely applied in various interactive scenarios due to its semantically-meaningful understanding and user-friendly controllability. However, existing 3D-morphable-model-based manipulation methods are not directly applicable to out-of-domain faces, such as non-photorealistic paintings, cartoon portraits, or even animals, mainly due to the formidable difficulties in building the model for each specific face domain. To overcome this challenge, we propose, as far as we know, the first method to manipulate faces in arbitrary domains using human 3DMM. This is achieved through two major steps: 1) disentangled mapping from 3DMM parameters to the latent space embedding of a pre-trained StyleGAN2 that guarantees disentangled and precise controls for each semantic attribute; and 2) cross-domain adaptation that bridges domain discrepancies and makes human 3DMM applicable to out-of-domain faces by enforcing a consistent latent space embedding. Experiments and comparisons demonstrate the superiority of our high-quality semantic manipulation method on a variety of face domains with all major 3D facial attributes controllable – pose, expression, shape, albedo, and illumination. Moreover, we develop an intuitive editing interface to support user-friendly control and instant feedback.



## To Do
- [ ] Release code


## Citation

If you find our work useful for your research, please consider citing the following papers :)

```
@article{wang2021crossdomain,
  title={Cross-Domain and Disentangled Face Manipulation with 3D Guidance},
  author={Wang, Can and Chai, Menglei and He, Mingming and Chen, Dongdong and Liao, Jing},
  journal={arXiv preprint arXiv:2104.11228},
  year={2021}
}
```
