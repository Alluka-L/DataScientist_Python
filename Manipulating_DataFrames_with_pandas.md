# Manipulating DataFrames with pandas

**Course Description**

>  In this course, you'll learn how to leverage pandas' extremely powerful data manipulation engine to get the most out of your data. You’ll learn how to drill into the data that really matters by extracting, filtering, and transforming data from DataFrames. The pandas library has many techniques that make this process efficient and intuitive. You will learn how to tidy, rearrange, and restructure your data by pivoting or melting and stacking or unstacking DataFrames. These are all fundamental next steps on the road to becoming a well-rounded data scientist, and you will have the chance to apply all the concepts you learn to real-world datasets.

## Extracting and transforming data

> In this chapter, you will learn how to index, slice, filter, and transform DataFrames using a variety of datasets, ranging from 2012 US election data for the state of Pennsylvania to Pittsburgh weather data.

### Index DataFrame

Pandas DataFrames provide several flexible means of indexing into subsets of our data.Let's start by importing pandas.And then read in a small DataFrame of sales data.

```python
import pandas as pd

df = pd.read_csv('sale.csv', indel_col='month')
df
```

|           | eggs | salt | spam |
| :-------- | :--- | ---- | :--- |
| **month** |      |      |      |
| Jan       | 47   | 12.0 | 17   |
| Feb       | 110  | 52.0 | 31   |
| Mar       | 221  | 89.0 | 72   |
| Apr       | 77   | 87.0 | 20   |
| May       | 132  | NaN  | 52   |
| Jun       | 205  | 60.0 | 55   |

The simplest indexing style uses   *`square brackets`*  , just as we would use to index Python lists and NumPy arrarys.

```python
df['salt']['Jan'] 
```

```python
12.0
```

Columns may also be referred to as  *`attributes`* of the DataFrame if their labels are valid Python identifiers.

```python
df.eggs['Mar']
```

```python
221
```

A more efficient and more programmatically redsable way of accessing data in  a DataFrame is by using accessors: `.loc`  and `.iloc` accessor.

The difference between them is that the fomer accesses by using labels, the latter using index positions. Both accessors use left bracket, row specifier, comma, column specifier, right bracket as syntax. 

For instance, here we  select spam sold in May using *`df.loc`*:

```python
df.loc['May', 'spam']
```

```python
52.0
```

On the other hand, here we select the same entry using *`df.iloc`*:

```python
df.iloc[4, 2]
```

```python
52.0
```

> *Remember lists and array-type structures in Python use zero-based indexing.*

When using bracket-indexing without the .loc or .iloc accessors, the result returned can be an individual value, a Pandas Series, or a  Pandas a DataFrame.

To ensure the return value is  a DataFrame, use a nested list within square brackets.

```python
df_new = df[['salt', 'eggs']]
df_new
```

|           | salt | eggs |
| --------- | ---- | ---- |
| **month** |      |      |
| Jan       | 12.0 | 47   |
| Feb       | 50.0 | 110  |
| Mar       | 89.0 | 221  |
| Apr       | 87.0 | 77   |
| May       | NaN  | 132  |
| Jun       | 60.0 | 205  |

> Notice we swapped the order of these two columns (relative to the ordering  in the original DataFrame) in the selection.

### Slicing DataFrames

Let's  look now at  slicing DataFrames. We're still using our sales DataFrame.

|           | eggs | salt | spam |
| :-------- | :--- | ---- | :--- |
| **month** |      |      |      |
| Jan       | 47   | 12.0 | 17   |
| Feb       | 110  | 52.0 | 31   |
| Mar       | 221  | 89.0 | 72   |
| Apr       | 77   | 87.0 | 20   |
| May       | 132  | NaN  | 52   |
| Jun       | 205  | 60.0 | 55   |

The basic indexing here picks a column by default. The result  returned  is  actully a pandas Series.

```python
df['egg']
```

```python
month
Jan     47
Feb    110
Mar    221
Apr     77
May    132
Jun    205
Name: eggs, dtype: int64
pandas.core.series.Series
```

A Series is a one-dimensional array with a labelled index (like a hybrid between a NumPy array and a dictionary).

Another way to think of a DataFrame is  a labelled two-dimensional array with Series for columns sharing common row labels.

Slicing can be performed with or without accessors.

```python
df['eggs'][1: 4]   # Part of the eggs column
```

```python
month
Feb.   110
Mar.   221
Apr.    77
Name: eggs, dtype: int64
```

```python
df['egg'][4]   # The value associated with May
```

```python
132
```

Pandas extends this colon syntax to allow labels in slices.

```python
df.loc[:, 'egg': 'salt']   # All rows, some columns
```

|           | eggs | salt |
| --------- | ---- | ---- |
| **month** |      |      |
| Jan       | 47   | 12.0 |
| Feb       | 110  | 50.0 |
| Mar       | 221  | 89.0 |
| Apr       | 77   | 87.0 |
| May       | 132  | NaN  |
| Jun       | 205  | 60.0 |

> Here, the first colon selects all rows. The slice `'eggs':'salt'` select both columns eggs and salt. This is a potential gotcha: slicing with labels and `.loc` accessor includes the right end-point (unlike positional slicing seen so far).

This example is similar in using `.loc` to slice all columns and some rows.

```python
df.loc['Jan': 'Apr', :]
```

|           | eggs | salt | spam |
| --------- | ---- | ---- | ---- |
| **month** |      |      |      |
| Jan       | 47   | 12.0 | 17   |
| Feb       | 110  | 50.0 | 31   |
| Mar       | 221  | 89.0 | 72   |
| Apr       | 77   | 87.0 | 20   |

> The first slice `'Jan': 'Apr'` extracts all four rows corresponding to January, February, March, and April inclusive. The second bare colon is a universal slice selecting all columns.

This example extracts a block with a proper subset of rows and columns.

```python
df.loc['Mar': 'May', 'salt': 'spam']
```

|           | salt | spam |
| --------- | ---- | ---- |
| **month** |      |      |
| Mar       | 89.0 | 72   |
| Apr       | 87.0 | 20   |
| May       | NaN  | 52   |

Using `.iloc` is very similar to `.loc` , simply with opsitional integers specifying slices rather than labels.

```python
df.iloc[2:5, 1:]   # A block from middle of the DataFrame
```

|           | salt | spam |
| --------- | ---- | ---- |
| **month** |      |      |
| Mar       | 89.0 | 72   |
| Apr       | 87.0 | 20   |
| May       | NaN  | 52   |

> From row 2 up to  but not  including row 5 and from  column 1 to the last column.

Both `.loc` and `.iloc` accessors can use `list` in place of slices.

```python
df.loc['Jan': 'May', '['eggs', 'spam']']
```

|           | eggs | spam |
| --------- | ---- | ---- |
| **month** |      |      |
| Jan       | 47   | 17   |
| Feb       | 110  | 31   |
| Mar       | 221  | 72   |
| Apr       | 77   | 20   |
| May       | 132  | 52   |

```python
df.iloc[[0, 4, 5], 0:2]
```

|           | eggs | salt |
| --------- | ---- | ---- |
| **month** |      |      |
| Jan       | 47   | 12.0 |
| May       | 132  | NaN  |
| Jun       | 205  | 60.0 |

>Remember, with `.iloc` the column slice 0:2 select two columns. 

Here's an important subtle distinction to understand:

``` python
df['xxx']    ->  pandas.core.series.Series
df[['xxx']]  ->  pandas.core.frame.DataFrame
```

$\color{red}{to\ be\ continued...}$

