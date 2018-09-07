# Lesson Proposal

In this lesson, I propose to use a spreadsheet and Python 3 based approach to look closely at how Term Frequency - Inverse Document Frequency (TF-IDF) works, when and why it is most often utilized, and how humanists--especially historians--can benefit from a better understanding of TF-IDF. AS the following lesson outline suggests, I'm planning to use a step-by-step approach that has three conceptual sections and one or two subsections per category. The general outline I imagine is as follows:

1. Preparation
  - Suggested Prior Skills  
  - TF-IDF Definition and Background
2. Procedure
  - How the Algorithm Works 
  - How to Run TF-IDF in Python 3
  - TF-IDF Compared with Alternative Techniques
3. Interpretation
  - Potential Variations of TF-IDF
  - Some Ways TF-IDF is Utilized in Humanities Scholarship
  - Interpreting Word Lists: Best Practices and Cautionary Notes

The central idea of this structure is that a better understanding of TF-IDF "under the hood" will help its users interpret the results more accurately and effectively. Additionally, the 'Procedure' section seeks to differentiate between (1) the mathematical manipulations of TF-IDF and (2) Scikit Learn's implementation of the algorithm (Python 3). 

Various sections will contain examples that the reader can interact with. At this point, I'm planning to use a corpus of 366 _New York Times_ historic obituaries that I scraped from https://archive.nytimes.com/www.nytimes.com/learning/general/onthisday/. It's a small enough dataset that readers can read some if not all of the files. Obituaries are a familiar genre to historians, and many of the historic figures are well known. Further, the _New York Times_ has begun publishing obituaries for overlooked women, so I think this choice of corpus will resonate with those invested in the politics of inclusion and representation. 


# Lesson Outline 

## 1. Preparation

### Suggested Prior Skills

I'm thinking that this tutorial should recommend some familiarity with Python or a similar programming language, as I plan to do the examples and interactive portions using Python 3. I'm also going to recommend familiarity with Excel or an equivalent spreadsheet application, which I think most readers will have.

### TF-IDF Definition and Background

This section will contain a definition, a bit of history on TF-IDF, and a specific example of TF-IDF in action.

## 2. Procedure

### How the Algorithm Works

This section will break down the TF-IDF operations step by step. There will be light math, but it's only multiplication and division in its simplest form. I'll also demonstrate with an Excel spreadsheet so it's easier to visualize what the "rows and columns" are.  

### How to Run it in Python 3

I plan to use the version in Scikit Learn for this in anticipation of potentially doing more lessons that focus on that particular library, partly because it's very useful in the humanities, and partly because its a very well written, well-documented library with lots of example pages.

### TF-IDF Compared with Alternative Techniques

TF-IDF can be compared with several other methods of "getting at" the meaningful term features in a collections of texts. It can also be contrasted with more sophisticated unsupervised sorting methods like topic modeling and 

### Potential Variations of TF-IDF

In this section, I want to discuss a Fivethirtyeight.com post from March 2016 called "These Are The Phrases Each GOP Candidate Repeats Most"(https://fivethirtyeight.com/features/these-are-the-phrases-each-gop-candidate-repeats-most/). It's a relatively straightforward post, but the visualization uses a modified TF-IDF that takes N-grams and performs the inverse-document frequency calculation on phrases rather than just words. I will walk readers through the process of adapting Fivethirtyeight's code to the obituary corpus I'm using in the rest of the tutorial. The result is, I think, interesting, and it demonstrates how the IDF operation can be extended. 

### Some Ways TF-IDF is Utilized in Humanities Scholarship

The idea of this section is to point to some scholarly uses of TF-IDF, as opposed to how it's used under the hood of a lot of everyday web applications.

- As an Exploratory Tool or Hermeneutic Aid
- As a Visualization Technique
- Searching for Similar Texts
- Making Feature Sets to Use with Machine Learning

### Interpreting Word Lists: Best Practices and Cautionary Notes

This section will attempt to generalize a bit, and will provide some concrete examples of how a TF-IDF term list can lead you in a direction that distorts your understanding of an underlying text. I will offer a few strategies designed to prevent the most common pitfalls.

- Cautionary Notes
- Generating Hypotheses or Research Questions
- Read at Least Some of the Underlying Texts 
- Test Robustness with Other Measures
- Following up with Direct Measures

### References

So far I know I want to use at least these sources and point people to them for more information: 

- Blei, David M., Andrew Y. Ng, and Michael I. Jordan, "Latent Dirichlet Allocation" _Journal of Machine Learning Research_ 3 (January 2003): 993-1022.
- Grimmer, Justin and King, Gary, Quantitative Discovery from Qualitative Information: A General-Purpose Document Clustering Methodology (2009). APSA 2009 Toronto Meeting Paper. Available at SSRN: https://ssrn.com/abstract=1450070
- Salton, G. and M.J. McGill, _Introduction to Modern Information Retrieval_. New York: McGraw-Hill, 1983.
Underwood, Ted. "Identifying diction that characterizes an author or genre: why Dunning’s may not be the best method," _The Stone and the Shell_, November 9, 2011. https://tedunderwood.com/2011/11/09/identifying-the-terms-that-characterize-an-author-or-genre-why-dunnings-may-not-be-the-best-method/
- --. "The Historical Significance of Textual Distances", Preprint of LaTeCH-CLfL Workshop, COLING, Santa Fe, 2018. https://arxiv.org/abs/1807.00181