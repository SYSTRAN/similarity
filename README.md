# similarity
Bilingual sentence similarity classifier based on word alignments optimization using Tensorflow

This repo implements a sentence similarity classifier model using Tensorflow. Similarity classification is based on the ideas introduced by [Carpuat et al., 2017](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjGyJLf5JHdAhXsz4UKHZcbBToQFjAAegQIARAC&url=http%3A%2F%2Faclweb.org%2Fanthology%2FW17-3209&usg=AOvVaw1aYY-B2TL4ZdVcc-zfY0hr) and [Vyas et al., 2018](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwj4v9L35JHdAhWryYUKHZueCw4QFjABegQICBAC&url=http%3A%2F%2Faclweb.org%2Fanthology%2FN18-1136&usg=AOvVaw2aWr7-b1Bkg3PiNXy1ZFed). The code borrows many of the concepts and architecture presented in [Legrand et al., 2016](http://www.aclweb.org/anthology/W16-2207).

The next picture, shows an example of similarity classification for a parallel sentence pair:

``` Hi , Maggie . Have you heard anything ? ||| As-tu entendu quelque chose ?```

<img src="https://github.com/jmcrego/divergence/blob/master/pics/divergence_example.png" width="250" />
