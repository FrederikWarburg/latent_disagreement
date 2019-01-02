# Investigation of the Neural Arithmetic Logic Unit

Contain the files, implementations and experiments conducted by the group Latent Disagreement during the 02456 Deep Learning course Fall 2018. Original paper: https://arxiv.org/abs/1808.00508

The NALU and NAC framework can be found within the "models" folder, whereas all the experiments are gathered in the "experiments" folder. These are to a large extend adapted from https://github.com/kevinzakka/NALU-pytorch. The "main_results" folder contains a script including the main findings during the project.

Experiments include:

* Subset selection and arithmetic operator (+, -, *, %)
* Only subset selection
* Only arithmetic operator

Furthermore, 3 possible extensions have been conducted:

* Temperature
* Learnable bias parameter
* Input-independent gating function

**Abstract**

Neural networks have proven advantageous in a wide variety of fields; including object detection, speech recognition, language processing. With a hunger for data, these networks can learn complex functions and interpolate really well. However, these networks tend to have problems when presented with data outside the training domain, often resulting in poor generalization capabilities. Recently, the Neural Arithemtic Logic Unit (NALU) has shown promising capabilities of extrapolating well beyond the training domain on numerous experiments. This paper seeks to investigate the NALU by replicating experiments from the original paper while exploring the potential problems this unit in prone to experience. Experiments show that the NALUs generally are hard to train, even when given simple problems such as addition of two numbers. In a variety of settings, it is seen how the presented gating function is likely to be the cause of poor extrapolation, especially given negative numbers. In some cases, however, the NALU shows promising results, outperforming traditional MLP's for tasks on positive numbers. Several experiments are conducted in order to train the model more smoothly, none of which showed great increase in stability.

Our paper can be found https://github.com/FrederikWarburg/latent_disagreement/blob/master/NALU_paper.pdf.

**General results**

Subset selection and arithemtic operation task

<img src="https://github.com/FrederikWarburg/latent_disagreement/blob/master/Images/FullTask.png" width="450">

Only subset selection task

<img src="https://github.com/FrederikWarburg/latent_disagreement/blob/master/Images/SubsetTask.png" width="450">

Propagation of weights. Here an example of convergence.

<img src="https://github.com/FrederikWarburg/latent_disagreement/blob/master/Images/kai_uni_conv.png" width="250">

In most cases however, the models had trouble learning the underlying structure, resulting in a poor generalization.

<img src="https://github.com/FrederikWarburg/latent_disagreement/blob/master/Images/kai_uni_div.png" width="250">

These results were most likely due to the fact that the gating function was hard to learn, meaning that the network "choose" the wrong gate and ended up multiplying instead of adding.

<img src="https://github.com/FrederikWarburg/latent_disagreement/blob/master/Images/kai_uni_div_g.png" width="250">

