


In the early weeks of the project, I implemented a basic set of functions capable of generating n-dimensional random values. Using these, I submitted random inputs for the first three weeks.

After gaining some familiarity with the problem, I modified my base function to implement the Upper Confidence Bound (UCB) acquisition strategy, which I intended to use for guiding input selection going forward.

Around week 20, I began experimenting with different kernel functions in my Gaussian Process model, aiming to better capture the structure of the function landscape. However, despite this effort, a critical mistake went unnoticed: my UCB implementation was incorrect. As a result, until 24, I was unknowingly submitting random inputs again—effectively reverting to my early approach without realizing it.

It wasn’t until week 24 that I identified this mistake. With just one week remaining in the competition, I pivoted to using the Expected Improvement (EI) strategy to guide input selection, making the most of the limited time left.




