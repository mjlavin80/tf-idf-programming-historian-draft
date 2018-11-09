# Processing, Exploring, and Analyzing a Document Corpus with TF-IDF

## Preparation

### Suggested Prior Skills

- Some familiarity with Python or a similar programming language. Code for this lesson is written in Python 3.6, but TF-IDF is available in many versions of Python and other programming languages.
- Familiarity with Excel or an equivalent spreadsheet application if you wish to examine the linked spreadsheet files.

### Before You Begin

- Install the Python 3 version of Anaconda. Installing Anaconda is covered in [Text Mining in Python through the HTRC Feature Reader](https://programminghistorian.org/en/lessons/text-mining-with-extracted-features). This will install Python 3.6 (or higher), the scikit-learn library (which we will use for TF-IDF), and the dependencies needed to run a Jupyter Notebook.   
- It is possible to install all these dependencies without Anaconda (or with a lightweight alternative like Miniconda). For more information, see the section below titled ["Alternatives to Anaconda"](#alternatives-to-anaconda)

### Lesson Dataset

TF-IDF, like many computational operations, is best understood by example. To this end, I've prepared a relatively small dataset of 366 _New York Times_ historic obituaries scraped from [https://archive.nytimes.com/www.nytimes.com/learning/general/onthisday/](https://archive.nytimes.com/www.nytimes.com/learning/general/onthisday/). For each day of the year, _The New York Times_ featured an obituary of someone born on that day. (There are 366 obituaries because of February 29 on the Leap Year.) The dataset is small enough that you should be able to open and read some if not all of the files. You'll notice that many of the historic figures are well known, which suggests a self-conscious effort to look back at the history of _The New York Times_ and select obituaries based on some criteria. In short, this isn't a representative sample of historic obituaries, it's a recent collection. 

This obituary corpus also an historical object in its own right. The version of _The New York Times_ "On This Day" website used for the dataset hasn't been updated since 2011, and it has been replaced by a newer, sleeker blog located at [https://learning.blogs.nytimes.com/on-this-day/](https://learning.blogs.nytimes.com/on-this-day/). It represents, on some level, how the questions inclusion and representation might affect both the decision to publish an obituary, and the decision to highlight a particular obituary many years later. The significance of such decisions has been further highlighted in recent months by _The New York Times_ itself. In march 2018, the newspaper began publishing obituaries for "overlooked women". In the words of Amisha Padnani and Jessica Bennett, "who gets remembered — and how — inherently involves judgment. To look back at the obituary archives can, therefore, be a stark lesson in how society valued various achievements and achievers." The dataset includes a central metada.csv file with each obituary's title and publication date. It also includes a folder of .html files downloaded from the 2011 "On This Day Website" and a folder of .txt files that represent the body of each obituary. These text files were generated using a Python library called BeautifulSoup, which is covered in another _Programming Historian_ (see [Intro to BeautifulSoup](https://programminghistorian.org/en/lessons/intro-to-beautiful-soup) ). The lesson files are located at [https://github.com/mjlavin80/tf-idf-programming-historian](https://github.com/mjlavin80/tf-idf-programming-historian). 

### TF-IDF Definition and Background

TF-IDF stands for Term Frequency - Inverse Document Frequency. Instead of representing a term in a document by its raw frequency (number of occurrences) or its relative frequency (term count divided by document length), each term is weighted by dividing the term frequency by the frequency of the word in the corpus, or the number of documents in the corpus containing the word. The overall effect of this weighting scheme is to avoid a common problem when conducting text analysis: the most frequently used words in a document are often the most frequently used words in all of the documents. Terms with the highest TF-IDF scores are the terms in a document that are _distinctly_ frequent in a  document, when that document is compared other documents. 

In a 1972 paper, Karen Spärck Jones explained the rationale for a weighting scheme based on term frequency weighted by collection frequency, and how it might be applied to information retrieval. Such weighting, she argued, "places greater emphasis on the value of a term as a means of distinguishing one document from another than on its value as an indication of the content of the document itself. ... In some cases a term may be common in a document and rare in the collection, so that it would be heavily weighted in both schemes. But the reverse may also apply. It is really that the emphasis is on different properties of terms." 

To understand how a term can be frequent but not distinct, or vice versa, let's look at an example. The following is a list of the top ten most frequent terms (and term counts) from one of the obituaries in our _New York Times_ corpus.

<div>
<table border="1" class="dataframe">
<thead>
	<tr style="text-align: right;">
		<th title="Rank">Rank</th>
	    <th title="Term">Term</th>
	    <th title="Count">Count</th>
    </tr>
</thead>
<tbody>
	<tr>
<td></td>
		<td>1</td>
		<td>the</td>
		<td>21</td>
	</tr>
	<tr>
<td></td>
		<td>2</td>
		<td>of</td>
		<td>16</td>
	</tr>
	<tr>
<td></td>
		<td>3</td>
		<td>her</td>
		<td>15</td>
	</tr>
	<tr>
<td></td>
		<td>4</td>
		<td>in</td>
		<td>14</td>
	</tr>
	<tr>
<td></td>
		<td>5</td>
		<td>and</td>
		<td>13</td>
	</tr>
	<tr>
<td></td>
		<td>6</td>
		<td>she</td>
		<td>10</td>
	</tr>
	<tr>
<td></td>
		<td>7</td>
		<td>at</td>
		<td>8</td>
	</tr>
	<tr>
<td></td>
		<td>8</td>
		<td>cochrane</td>
		<td>4</td>
	</tr>
	<tr>
<td></td>
		<td>9</td>
		<td>was</td>
		<td>4</td>
	</tr>
	<tr>
<td></td>
		<td>10</td>
		<td>to</td>
		<td>4</td>
	</tr>
</tbody>
</table>
</div>

After looking at this list, imagine trying to discern information about the obituary that this table represents. We might infer from the presence of "her" and "cochrane" in the list that a woman named Cochrane is being discussed but, at the same time, this could easily be about a person from Cochrane, Wisconsin or someone associated with the Cochrane Collaboration, a non-profit, non-governmental organization. The problem with this list is that most of top terms would be top terms in any obituary and, indeed, any sufficiently large chunk of writing in most languages. This is because most languages are heavily dependent on function words like "the," "as," "of," "to," and "from" that serve primarily grammatical or structural purposes, and appear regardless of the text's subject matter. A list of an obituary's most frequent terms tell us little about the obituary or the person being memorialized.  Now let's use TF-IDF term weighting to compare the same obituary from the first example to the rest of our corpus of _New York Times_ obituaries. The top ten term scores look like this: 

<div>
<table border="1" class="dataframe">
<thead>
	<tr style="text-align: right;">
		<th title="Rank">Rank</th>
	    <th title="Term">Term</th>
	    <th title="Count">Count</th>
    </tr>
</thead>
<tbody>
	<tr>
<td></td>
		<td>1</td>
		<td>cochrane</td>
		<td>24.85</td>
	</tr>
	<tr>
<td></td>
		<td>2</td>
		<td>her</td>
		<td>22.74</td>
	</tr>
	<tr>
<td></td>
		<td>3</td>
		<td>she</td>
		<td>16.22</td>
	</tr>
	<tr>
<td></td>
		<td>4</td>
		<td>seaman</td>
		<td>14.88</td>
	</tr>
	<tr>
<td></td>
		<td>5</td>
		<td>bly</td>
		<td>12.42</td>
	</tr>
	<tr>
<td></td>
		<td>6</td>
		<td>nellie</td>
		<td>9.92</td>
	</tr>
	<tr>
<td></td>
		<td>7</td>
		<td>mark</td>
		<td>8.64</td>
	</tr>
	<tr>
<td></td>
		<td>8</td>
		<td>ironclad</td>
		<td>6.21</td>
	</tr>
	<tr>
<td></td>
		<td>9</td>
		<td>plume</td>
		<td>6.21</td>
	</tr>
	<tr>
<td></td>
		<td>10</td>
		<td>vexations</td>
		<td>6.21</td>
	</tr>
</tbody>
</table>

In this version of the list, "she" and "her" have both moved up. "cochrane" remains, but now we have at least two new name-like words: "nellie" and "nly." Nellie Bly, of course, was a turn-of-the-century journalist. She was born Elizabeth Cochrane Seaman, and Bly was her pen name or _nom-de-plume_. With only these sparse details, we now account for seven of our ten terms: "cochrane," "her," "she," "seaman," "bly," "nellie," and "plume." To understand "mark," "ironclad," and "vexations," we can return to the original obituary and discover that Bly died at St. Mark's Hospital. Her husband was president of the Ironclad Manufacturing Company. Finally, "a series of forgeries by her employees, disputes of various sorts, bankruptcy and a mass of vexations and costly litigations swallowed up Nellie Bly's fortune." Many of the terms on this list are mentioned as few as one, two, or three times; they are not frequent by any measure. Their presence in this one document, however, are all distinct compared with the rest of the corpus. 

## Procedure 

### How the Algorithm Works

TF-IDF can be implemented in many flavors, some more complex than others. Before I begin discussing these complexities, however, I would like to trace the algorithmic operations of one particular version. To this end, we will go back to the Nellie Bly obituary and convert the top ten term counts into TF-IDF scores using the same steps that were used to create the above TF-IDF example. These steps parallel scikit learn's TF-IDF implementation. 


Addition, multiplication, and division are the primary mathematical operations necessary to follow along. At one point, we must perform calculate the natural logarithm of a variable, but this can be done with most online calculators and calculator mobile apps. (You can also download an Excel spreadsheet that represents the operations for all 206 terms in the Bly obituary.) Below is a table with the raw term counts for the first thirty words, in alphabetical order, from Bly's obituary, but this version has a second column that represents the number of documents in which each term can be found.

<div>
<table border="1" class="dataframe">
<thead>
	<tr style="text-align: right;">
		<th title="Term">Term</th>
		<th title="Count">Count</th>
		<th title="DF">DF</th>
		
    </tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>afternoon</td>
<td>1</td>
<td>66</td>
</tr>
<tr>
<td>2</td>
<td>against</td>
<td>1</td>
<td>189</td>
</tr>
<tr>
<td>3</td>
<td>age</td>
<td>1</td>
<td>224</td>
</tr>
<tr>
<td>4</td>
<td>ago</td>
<td>1</td>
<td>161</td>
</tr>
<tr>
<td>5</td>
<td>air</td>
<td>1</td>
<td>80</td>
</tr>
<tr>
<td>6</td>
<td>all</td>
<td>1</td>
<td>310</td>
</tr>
<tr>
<td>7</td>
<td>american</td>
<td>1</td>
<td>277</td>
</tr>
<tr>
<td>8</td>
<td>an</td>
<td>1</td>
<td>352</td>
</tr>
<tr>
<td>9</td>
<td>and</td>
<td>13</td>
<td>364</td>
</tr>
<tr>
<td>10</td>
<td>around</td>
<td>2</td>
<td>149</td>
</tr>
<tr>
<td>11</td>
<td>as</td>
<td>2</td>
<td>357</td>
</tr>
<tr>
<td>12</td>
<td>ascension</td>
<td>1</td>
<td>6</td>
</tr>
<tr>
<td>13</td>
<td>asylum</td>
<td>1</td>
<td>2</td>
</tr>
<tr>
<td>14</td>
<td>at</td>
<td>8</td>
<td>362</td>
</tr>
<tr>
<td>15</td>
<td>avenue</td>
<td>2</td>
<td>68</td>
</tr>
<tr>
<td>16</td>
<td>balloon</td>
<td>1</td>
<td>2</td>
</tr>
<tr>
<td>17</td>
<td>bankruptcy</td>
<td>1</td>
<td>8</td>
</tr>
<tr>
<td>18</td>
<td>barrel</td>
<td>1</td>
<td>7</td>
</tr>
<tr>
<td>19</td>
<td>baxter</td>
<td>1</td>
<td>4</td>
</tr>
<tr>
<td>20</td>
<td>be</td>
<td>1</td>
<td>332</td>
</tr>
<tr>
<td>21</td>
<td>beat</td>
<td>1</td>
<td>33</td>
</tr>
<tr>
<td>22</td>
<td>began</td>
<td>1</td>
<td>241</td>
</tr>
<tr>
<td>23</td>
<td>bell</td>
<td>1</td>
<td>24</td>
</tr>
<tr>
<td>24</td>
<td>bly</td>
<td>2</td>
<td>1</td>
</tr>
<tr>
<td>25</td>
<td>body</td>
<td>1</td>
<td>112</td>
</tr>
<tr>
<td>26</td>
<td>born</td>
<td>1</td>
<td>342</td>
</tr>
<tr>
<td>27</td>
<td>but</td>
<td>1</td>
<td>343</td>
</tr>
<tr>
<td>28</td>
<td>by</td>
<td>3</td>
<td>349</td>
</tr>
<tr>
<td>29</td>
<td>career</td>
<td>1</td>
<td>223</td>
</tr>
<tr>
<td>30</td>
<td>character</td>
<td>1</td>
<td>89</td>
</tr>
</tbody>
</table>
</div>

The document frequency (DF) is no more than a count of how many documents from the corpus each word appears in. To convert these document counts, to an inverse document frequency, the most direct formula would be N/DF, where N represents the total number of documents in the corpus. However, many implementations (including the original TF-IDF implementation) normalize the results with additional operations. For example, scikit-learn's implementation calculates N as N+1, then calculates the natural logarithm of (N+1)/DF, and then adds 1 to the final result. To summarize this IDF equation, then: 

IDF = ln[(N+1)/DF] + 1



<div>
<table border="1" class="dataframe">
<thead>
	<tr style="text-align: right;">
		<th title="Term">Term</th>
		<th title="Count">Count</th>
		<th title="DF">DF</th>
		<th title="Smoothed-IDF">Smoothed-IDF</th>
		<th title="TF-IDF">TF-IDF</th>
		<th title="Comments">Comments</th>
    </tr>
</thead>
<tbody>
<tr>
<td>1</td>
<td>afternoon</td>
<td>1</td>
<td>66</td>
<td>2.70066923</td>
<td>2.70066923</td>
<td></td>
</tr>
<tr>
<td>2</td>
<td>against</td>
<td>1</td>
<td>189</td>
<td>1.65833778</td>
<td>1.65833778</td>
<td> </td>
</tr>
<tr>
<td>3</td>
<td>age</td>
<td>1</td>
<td>224</td>
<td>1.48926145</td>
<td>1.48926145</td>
<td> </td>
</tr>
<tr>
<td>4</td>
<td>ago</td>
<td>1</td>
<td>161</td>
<td>1.81776551</td>
<td>1.81776551</td>
<td> </td>
</tr>
<tr>
<td>5</td>
<td>air</td>
<td>1</td>
<td>80</td>
<td>2.51091269</td>
<td>2.51091269</td>
<td> </td>
</tr>
<tr>
<td>6</td>
<td>all</td>
<td>1</td>
<td>310</td>
<td>1.16556894</td>
<td>1.16556894</td>
<td>DF over max</td>
</tr>
<tr>
<td>7</td>
<td>american</td>
<td>1</td>
<td>277</td>
<td>1.27774073</td>
<td>1.27774073</td>
<td>DF over max</td>
</tr>
<tr>
<td>8</td>
<td>an</td>
<td>1</td>
<td>352</td>
<td>1.03889379</td>
<td>1.03889379</td>
<td>DF over max</td>
</tr>
<tr>
<td>9</td>
<td>and</td>
<td>13</td>
<td>364</td>
<td>1.00546449</td>
<td>13.07103843</td>
<td>DF over max</td>
</tr>
<tr>
<td>10</td>
<td>around</td>
<td>2</td>
<td>149</td>
<td>1.89472655</td>
<td>3.78945311</td>
<td></td>
</tr>
<tr>
<td>11</td>
<td>as</td>
<td>2</td>
<td>357</td>
<td>1.02482886</td>
<td>2.04965772</td>
<td>DF over max</td>
</tr>
<tr>
<td>12</td>
<td>ascension</td>
<td>1</td>
<td>6</td>
<td>4.95945170</td>
<td>4.95945170</td>
<td></td>
</tr>
<tr>
<td>13</td>
<td>asylum</td>
<td>1</td>
<td>2</td>
<td>5.80674956</td>
<td>5.80674956</td>
<td></td>
</tr>
<tr>
<td>14</td>
<td>at</td>
<td>8</td>
<td>362</td>
<td>1.01095901</td>
<td>8.08767211</td>
<td>DF over max</td>
</tr>
<tr>
<td>15</td>
<td>avenue</td>
<td>2</td>
<td>68</td>
<td>2.67125534</td>
<td>5.34251069</td>
<td></td>
</tr>
<tr>
<td>16</td>
<td>balloon</td>
<td>1</td>
<td>2</td>
<td>5.80674956</td>
<td>5.80674956</td>
<td></td>
</tr>
<tr>
<td>17</td>
<td>bankruptcy</td>
<td>1</td>
<td>8</td>
<td>4.70813727</td>
<td>4.70813727</td>
<td></td>
</tr>
<tr>
<td>18</td>
<td>barrel</td>
<td>1</td>
<td>7</td>
<td>4.82592031</td>
<td>4.82592031</td>
<td></td>
</tr>
<tr>
<td>19</td>
<td>baxter</td>
<td>1</td>
<td>4</td>
<td>5.29592394</td>
<td>5.29592394</td>
<td></td>
</tr>
<tr>
<td>20</td>
<td>be</td>
<td>1</td>
<td>332</td>
<td>1.09721936</td>
<td>1.09721936</td>
<td>DF over max</td>
</tr>
<tr>
<td>21</td>
<td>beat</td>
<td>1</td>
<td>33</td>
<td>3.37900132</td>
<td>3.37900132</td>
<td></td>
</tr>
<tr>
<td>22</td>
<td>began</td>
<td>1</td>
<td>241</td>
<td>1.41642412</td>
<td>1.41642412</td>
<td>DF over max</td>
</tr>
<tr>
<td>23</td>
<td>bell</td>
<td>1</td>
<td>24</td>
<td>3.68648602</td>
<td>3.68648602</td>
<td></td>
</tr>
<tr>
<td>24</td>
<td>bly</td>
<td>2</td>
<td>1</td>
<td>6.21221467</td>
<td>12.42442933</td>
<td></td>
</tr>
<tr>
<td>25</td>
<td>body</td>
<td>1</td>
<td>112</td>
<td>2.17797403</td>
<td>2.17797403</td>
<td></td>
</tr>
<tr>
<td>26</td>
<td>born</td>
<td>1</td>
<td>342</td>
<td>1.06763140</td>
<td>1.06763140</td>
<td>DF over max</td>
</tr>
<tr>
<td>27</td>
<td>but</td>
<td>1</td>
<td>343</td>
<td>1.06472019</td>
<td>1.06472019</td>
<td>DF over max</td>
</tr>
<tr>
<td>28</td>
<td>by</td>
<td>3</td>
<td>349</td>
<td>1.04742869</td>
<td>3.14228608</td>
<td>DF over max</td>
</tr>
<tr>
<td>29</td>
<td>career</td>
<td>1</td>
<td>223</td>
<td>1.49371580</td>
<td>1.49371580</td>
<td></td>
</tr>
<tr>
<td>30</td>
<td>character</td>
<td>1</td>
<td>89</td>
<td>2.40555218</td>
<td>2.40555218</td>
<td></td>
</tr>
</tbody>
</table>
</div>


### How to Run it in Python 3

Scikit-Learn version

There are several Python libraries that include 

### TF-IDF Compared with Alternative Techniques

TF-IDF can be compared with several other methods of "getting at" the meaningful term features in a collections of texts. It can also be contrasted with more sophisticated unsupervised sorting methods like topic modeling and clustering.

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

## References

- Beckman, Milo. "These Are The Phrases Each GOP Candidate Repeats Most," _FiveThirtyEight_, March 10, 2016. https://fivethirtyeight.com/features/these-are-the-phrases-each-gop-candidate-repeats-most/
- Bennett, Jessica, and Amisha Padnani. "Overlooked," March 8, 2018. https://www.nytimes.com/interactive/2018/obituaries/overlooked.html
- Blei, David M., Andrew Y. Ng, and Michael I. Jordan, "Latent Dirichlet Allocation" _Journal of Machine Learning Research_ 3 (January 2003): 993-1022.
- Grimmer, Justin and King, Gary, Quantitative Discovery from Qualitative Information: A General-Purpose Document Clustering Methodology (2009). APSA 2009 Toronto Meeting Paper. Available at SSRN: https://ssrn.com/abstract=1450070
- Salton, G. and M.J. McGill, _Introduction to Modern Information Retrieval_. New York: McGraw-Hill, 1983.
- Sparck Jones, Karen. "A Statistical Interpretation of Term Specificity and Its Application in Retrieval." Journal of Documentation 28, no. 1 (1972): 11–21.
- Underwood, Ted. "Identifying diction that characterizes an author or genre: why Dunning’s may not be the best method," _The Stone and the Shell_, November 9, 2011. https://tedunderwood.com/2011/11/09/identifying-the-terms-that-characterize-an-author-or-genre-why-dunnings-may-not-be-the-best-method/
- --. "The Historical Significance of Textual Distances", Preprint of LaTeCH-CLfL Workshop, COLING, Santa Fe, 2018. https://arxiv.org/abs/1807.00181

## Alternatives to Anaconda

If you are not using Anaconda, you will need to cover the following dependencies:

1. Install Python 2 or 3 (preferably Python 3.6 or later)
2. Recommended: install and run a virtual environment
3. Install the Scikit-Learn library and its dependencies (see [http://scikit-learn.org/stable/install.html](http://scikit-learn.org/stable/install.html)).
4. Install Jupyter Notebook and its dependencies