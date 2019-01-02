# Investigation of the NALU

Contain the files, implementations and experiments conducted by the group Latent Disagreement during the 02456 Deep Learning course Fall 2018.

The NALU and NAC framework can be found within the "models" folder, whereas all the experiments are gathering in the "experiments" folder. 

**Abstract**

Neural networks have proven advantageous in a wide variety of fields; including object detection, speech recognition, language processing. With a hunger for data, these networks can learn complex functions and interpolate really well. However, these networks tend to have problems when presented with data outside the training domain, often resulting in poor generalization capabilities. Recently, the Neural Arithemtic Logic Unit (NALU) has shown promising capabilities of extrapolating well beyond the training domain on numerous experiments. This paper seeks to investigate the NALU by replicating experiments from the original paper while exploring the potential problems this unit in prone to experience. Experiments show that the NALUs generally are hard to train, even when given simple problems such as addition of two numbers. In a variety of settings, it is seen how the presented gating function is likely to be the cause of poor extrapolation, especially given negative numbers. In some cases, however, the NALU shows promising results, outperforming traditional MLP's for tasks on positive numbers. Several experiments are conducted in order to train the model more smoothly, none of which showed great increase in stability.

**General results**

