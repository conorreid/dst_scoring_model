#D/ST scoring model

### Summary
This is D/ST fantasy football scoring model, using a model framework loosely 
based on that of quickonthedrawl in the r/fantasyfootball reddit subreddit
(detailed here: https://www.reddit.com/r/fantasyfootball/comments/9cv5of/building_and_improving_on_existing_dst_projections/). 

### Sources
- teamrankings.com for team statistics
- Pinnacle for betting lines

### Methodology
As outlined in post, creates predicted D/ST touchdowns, points allowed, sacks,
and interceptions using a combination of opponent and team statistics over both
all of last season and the last 3 games of this season. It then uses these 
average values and creates a Poisson distribution with each against a range of 
zero occurances of that event with a max likely decided by myself. These 
predicted occurances are then multiplied by the scores they would recieve to 
calculated the final predicted score of defense for that week. 