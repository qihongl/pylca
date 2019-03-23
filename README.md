# The leaky, competing, accumulator (LCA)

this is a lightweight python implementation of the leaky, competing, accumulator, based on [1], [2] and [3]. The default is to behave like [2]. 

<br>

#### how to use

here's an 
<a href="https://github.com/qihongl/pylca/tree/master/example">example</a> 
that you can play with on google colab: <a href="https://colab.research.google.com/github/qihongl/pylca/blob/master/example/demo_lca.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory"></a>

#### install 

its on 
<a href="https://pypi.org/project/pylca/">PyPI</a>, 
so simply do `pip install pylca`


#### note 

this implementation... 
- ... allows any non-negative self-excitation. [1, 2] assume self-excitation of the accumulators is zero. 
- ... doesn't terminate the LCA process when the (activity threshold) criterion is met, which is different from [2]. the user can truncate the activity time course post-hoc. 
- ... lower bound the output activity by 0 (i.e. ReLU), like [1, 2]. [3] can do other non-linear transformations
- ... doesn't perform exponential weighted moving average of the inputs. [3] can do this. 


#### references:  

- [1] Usher, M., & McClelland, J. L. (2001). The time course of perceptual choice: the leaky, competing accumulator model. Psychological Review, 108(3), 550–592. Retrieved from https://www.ncbi.nlm.nih.gov/pubmed/11488378

- [2] Polyn, S. M., Norman, K. A., & Kahana, M. J. (2009). A context maintenance and retrieval model of organizational processes in free recall. Psychological Review, 116(1), 129–156. https://doi.org/10.1037/a0014420 

- [3] <a href="https://github.com/PrincetonUniversity/PsyNeuLink">PsyNeuLink</a>: <a href="https://princetonuniversity.github.io/PsyNeuLink/LCAMechanism.html">LCAMechanism</a>
