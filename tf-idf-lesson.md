# Suggested Prior Skills

- Some familiarity with Python or a similar programming language
- Familiarity with Excel or an equivalent spreadsheet application

# TF-IDF Definition and Background

TF-IDF stands for Term Frequency - Inverse Document Frequency. Instead of representing a term in a document by its raw frequency (number of occurrences) or its relative frequency (term count divided by document length), each term is weighted by dividing the term frequency by the frequency of the word in the corpus, or the number of documents in the corpus containing the word. The overall effect of this weighting scheme is to avoid a common problem when conducting text analysis: the most frequently used words in a document are often the most frequently used words in all of the documents. Terms with the highest TF-IDF scores are the terms in a document that are _distinctly_ frequent in a  document, when that document is compared other documents. 

Consider the following example:

# TF-IDF Compared with Alternative Techniques

use obits from nytimes: bly, tarbell, baker, riis

# How the Algorithm Works

counts for four documents, N features: 
run first in Python to use as a basis
converted to freqs
log value conversion
doc freq column
final result
influence of numerator and denominator - rare vs. common words

# Other Potential Uses 

fivethirtyeight N-gram-idf

# Some Ways TF-IDF is Used

- ## As an Exploratory Tool
- ## As a Visualization Technique
- ## Searching for Similar Texts
- ## Making Feature Sets
- ## Machine Learning

# Interpreting Word Lists

- ## Cautionary Notes
- ## Generating Hypotheses or Research Questions
- ## Following up with Direct Measures

# References

Blei, David M., Andrew Y. Ng, and Michael I. Jordan, "Latent Dirichlet Allocation" _Journal of Machine Learning Research_ 3 (January 2003): 993-1022.
Salton, G. and M.J. McGill, _Introduction to Modern Information Retrieval_. New York: McGraw-Hill, 1983.
