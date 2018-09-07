




# Suggested Prior Skills

I'm thinking that this tutorial should recommend some familiarity with Python or a similar programming language, as I plan to do the examples and interactive portions using Python 3. I'm also going to recommend familiarity with Excel or an equivalent spreadsheet application, which I think most readers will have.

# TF-IDF Definition and Background

This section will contain a definition, a bit of history on TF-IDF, and a specific example of TF-IDF in action.

# TF-IDF Compared with Alternative Techniques

TF-IDF can be compared with several other methods of "getting at" the meaningful term features in a collections of texts. It can also be contrasted with more sophisticated unsupervised sorting metods like topic modeling and 

# How the Algorithm Works

This section will break down the TF-IDF operations step by step. There will be light math, but it's only multiplication and division in its simplest form. I'll also demonstrate with an Excel spreadsheet so it's easier to visualize what the "rows and columns" are.  

# Other Potential Uses 

In this section, I want to reproduce/discuss a fivethirtyeight.com post from March 2016 called "These Are The Phrases Each GOP Candidate Repeats Most"(https://fivethirtyeight.com/features/these-are-the-phrases-each-gop-candidate-repeats-most/). It's a relatively straightforward post, but the visualization uses a modified TF-IDF that takes N-grams and performs the inverse-document frequency calculation on phrases rather than just words. The result is, I think, interesting, and its shows that you can do the IDF operation on things other than term counts. 

# Some Ways TF-IDF is Used

The idea of this section is to point to some scholarly uses of TF-IDF, as opposed to how it's used under the hood of a lot of everyday web applications.

- ## As an Exploratory Tool or Hermeneutic Aid
- ## As a Visualization Technique
- ## Searching for Similar Texts
- ## Making Feature Sets to use with Machine Learning

# Interpreting Word Lists

This section will attempt to generalize a bit, and will provide some concrete examples of how a TF-IDF term list can lead you in a direction that distorts your understanding of an underlying text. I will offer a few strategies designed to prevent the most common pitfalls.

- ## Cautionary Notes
- ## Generating Hypotheses or Research Questions
- ## Read at Least Some of the Underlying Texts 
- ## Test Robustness with Other Measures
- ## Following up with Direct Measures

# References

So far I know I want to use at least these sources and point people to them for more information: 

- Blei, David M., Andrew Y. Ng, and Michael I. Jordan, "Latent Dirichlet Allocation" _Journal of Machine Learning Research_ 3 (January 2003): 993-1022.
- Grimmer, Justin and King, Gary, Quantitative Discovery from Qualitative Information: A General-Purpose Document Clustering Methodology (2009). APSA 2009 Toronto Meeting Paper. Available at SSRN: https://ssrn.com/abstract=1450070
- Salton, G. and M.J. McGill, _Introduction to Modern Information Retrieval_. New York: McGraw-Hill, 1983.
Underwood, Ted. "Identifying diction that characterizes an author or genre: why Dunning’s may not be the best method," _The Stone and the Shell_, November 9, 2011. https://tedunderwood.com/2011/11/09/identifying-the-terms-that-characterize-an-author-or-genre-why-dunnings-may-not-be-the-best-method/
- --. "The Historical Significance of Textual Distances", Preprint of LaTeCH-CLfL Workshop, COLING, Santa Fe, 2018. https://arxiv.org/abs/1807.00181