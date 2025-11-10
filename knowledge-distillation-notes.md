
1. [Basic knowledge distillation](#basic-knowledge-distillation--)
2. [2 stage Multi Teacher Knowledge Distillation (TMKD)](#2-stage-multi-teacher-knowledge-distillation-tmkd--)

## Teacher and student networks –
* Knowledge distillation – Capture and distill knowledge learned by model in a more compact model by using knowledge distillation training method
* It aims to create a more efficient model which captures the same knowledge as a more sophisticated and complex model. If needed, further optimization can be applied to the result.
* It is a way to train a small model to mimic a larger model or even an ensemble of models.
    * It starts by 1st training a complex model or a model ensemble and achieve high accuracy.
    * It then uses that model as a teacher for a simpler student model which will be the actual model that gets deployed in production
    * This teacher network can either be fixed or jointly optimized and can even be used to train multiple student models of different sizes simultaneously.

### Basic knowledge distillation -

**Training objectives of teacher and student models –**
1.	Teacher – normal training (maximizes actual metric)
2.	Student (training for knowledge transfer)
   
**Training objective of loss function –**
* Aims to match the prob distribution of teacher’s predictions i.e., the targets while computing loss function is the distribution of class probs predicted by the teacher model. These probabilities of predictions form soft targets.
* Note that the student is not only learning the teacher’s predictions, but also the probs of the predictions
* Soft targets tell us more about the knowledge about the teacher than the predictions alone.

**How does knowledge distillation work?**
* During knowledge distillation, the logits by teacher model are fed to softmax function to produce soft targets (probs of predictions) as they produce more info about the probabilities of all target classes for each example. However, in many cases, this probability distribution has correct class with high probability with all the rest of the class probabilities very close to 0. This is due to softmax function’s tendency to push high values toward 1 and low values toward 0
* So practically, it sometimes doesn’t provide much info beyond the ground truth labels already provided in the dataset.
* To tackle this, tweak softmax Temperature parameter T in the objective of teacher and student models during distillation.
* Increasing the temp T in objective of student and teacher, increases the softness of the teacher’s distribution as it changes the shape of the distribution towards being more uniform (rather than being sharp), leading to more diverse output (probability is more spread around the vocab)
* While a lower temperature would make the probability distribution more sharp, leading to less diverse output (probability is concentrated on top words) (ref – [Stanford ML](https://cs229.stanford.edu/main_notes.pdf),
  [Stanford – Neural Language Generation](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture10-nlg.pdf))
* Thus, as T grows, we get more insight about which classes teacher finds similar to the predicted one

  $p_i = \frac{exp(z_i/T)}{\sum_j{exp(z_i/T)}}$

2 types of knowledge distillation techniques –
1.	Weigh – combine both soft targets and hard labels
2.	Compare distribution of predictions (student and teacher) using KL divergence (more widely used)

2nd technique - using KL divergence –

* With this, knowledge distillation is performed by blending two loss functions, where alpha’s value is chosen between 0 and 1.

$L = (1- \alpha) L_H + \alpha L_KL$

$L$ – CEL (Cross entropy loss) from hard labels
$L_KL$ = KL divergence loss from teacher’s logits

* KL divergence here is the metric of the difference between 2 prob distributions
* Objective here is to make the distribution over the classes predicted by student as close as possible to the teacher.

* For computing distillation loss, same value for temperature T is used for both teacher and student in softmax.
* Based on the research, it has been found that distilled models also predict correct labels in addition to the soft targets by teacher.
* Thus, knowledge distillation also includes a standard loss, between the student’s predicted class probs and the ground truth labels. These are called hard labels / targets. This loss is student loss.
* So, when calculating probs for student loss, max T is set to 1.

* The quantitative results of distillation shows that distillation can extract more useful information from the training set than merely using the hard labels to train a single model

    * E.g., DistilBERT – uses 40% fewer params, runs 60% faster, while preserving 97 % accuracy of teacher model.
    * Smaller version of BERT, where the token type embeddings and pooler layer used for Next sentence classification task are removed. The rest of the architecture remains identical, while reducing the no. of layers.


References -
* [Stanford ML](https://cs229.stanford.edu/main_notes.pdf)
* [Stanford – Neural Language Generation](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1234/slides/cs224n-2023-lecture10-nlg.pdf)
* https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production/home/module/3
* https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
  

### 2 stage Multi Teacher Knowledge Distillation (TMKD) -
Case study - knowledge distillation for a Q&A task
**Motivation for TMKD approach -**  
* Applying complex models to real business scenarios become challenging due to the vast amount of model parameters.
* Older model compression methods suffer from information loss during the model compression procedure.
* This leads to inferior models compared to the original one.
* This basic knowledge distillation approach is known as a one-on-one model because one teacher transfers knowledge to one student
* Although this approach effectively reduces the number of parameters and the time for model inference, due to the information loss during knowledge desolation, the performance of the student model is sometimes not on par with that of the teacher.
 
**Approach to tackle this limitation - two stage multi teacher knowledge distillation method (TMKD) -**
**Gist -**
1. First develop a general Q&A distillation task for student model pre training. 
2.	And further fine tune this pretrained student model with a multi teacher knowledge distillation model.
    * This approach is called M on M or many on many ensemble model. Combining both ensemble and knowledge distillation. 

**Approach -**
* This involves first training multiple teacher models. The models could be BERT or GPT or others similarly powerful models, each having different hyper-parameters.
* Then a student model for each teacher model is then trained.
* Finally, the student models trained from different teachers are ensemble to create the final result. Here, you prepare each teacher for a particular learning objective and then train them.
* Different models have different generalization of capabilities and they also over fit the training data in different ways, achieving performance close to the teacher model.
* TMKD outperforms various state of the art baselines and it has been applied to real commercial scenarios since ensemble it is employed here.
* These compressed models benefit from large scale data and learned feature representations well.
* Results from experiments show that it can considerably outperformed baseline methods. And even achieve comparable results to the original teacher models along with a substantial speed up of model inference. 

**Benefits of TMKD -**
* TMKD uses a multi teacher distillation task for student model pre training to boost model performance.  
    * Based on experimental evaluations, MKD outperforms KD on a majority of tasks demonstrating that a multi teacher distillation approach can help the student model learn more generalized knowledge, fusing knowledge from different teachers.
    * Further, TMKD significantly outperforms TKD at MKD in all datasets, which verifies the complementary impact of the two stages distillation: pre training and multi teacher fine tuning. 


References -
* https://www.coursera.org/learn/machine-learning-modeling-pipelines-in-production/home/module/3
* [Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System, Yang et al., ACM 2020](https://arxiv.org/pdf/1910.08381)





