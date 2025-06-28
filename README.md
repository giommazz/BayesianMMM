# Use Bayesian Mixed-Media Model to decide marketing actions
- Context: performance marketing
- Goal: build a bayesian mixed-media model (MMM) to examine the performance of different marketing channels, then interpret the insights from the model
- How: use the latest PyMC package (https://www.pymc.io/) to code the MMM 


## Context
A company owns an online shop and advertises on seven different paid marketing channels (say, TV, radio, billboards, Google Ads, Facebook Ads, etc.), each with weekly costs.
Marketing expenditures often have a delayed effect on sales, as ads and campaigns in one week usually have an impact only on sales in the next weeks. 
Since different channels can be expected to target different audiences at different times, they might have diverse impact on future sales.
The company is then interested in understanding how its effective different channels are, and want to build a probabilistic model capable of modelling the uncertainty and the delayed effects of its marketing actions.


## Tasks
* Model the spend carry over effect (_adstock_);
* Include _seasonality_ and _trend_ in the MMM model;
* Bonus: consider using _saturation_ or _diminishing returns_ to model adstock shape effects
* Explain which _prior inputs_ were used in the MMM;
* Examine the model results, comparing _prior sampling_ VS _posterior sampling_;
* Measure model performance;
* Explain main insights in terms of channel performance and effects;
* Derive return on investment (ROI) estimates per channel, and identify the best channel in terms of ROI.


## Dataset: `MMM_test_data.csv`
Column info:
* `start_of_week`: first day of the week	
* `revenue`: sales revenue generated in current week	
* `spend_channel_1...7`: marketing spend in current week in channel 1...7	
