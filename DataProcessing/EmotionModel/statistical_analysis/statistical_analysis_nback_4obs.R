# # installs
# install.packages("lme4")
# install.packages("lmerTest")
# install.packages("dplyr")
# install.packages("emmeans")
# install.packages("pbkrtest")
# install.packages("multcomp")
# install.packages("nlme")
# install.packages("Crossover")
# install.packages("rJava")
# install.packages("plotly")
# install.packages("sjPlot")

# package imports
library(lme4)
library(ggplot2)
library(sjmisc)
library(sjPlot)
library(plotly)
library(rJava)
library(Crossover)
library(lmerTest)
library(dplyr)
library(emmeans)
library(pbkrtest)
library(multcomp)
library(gridExtra)
library(nlme)

# clear enviornment of variables
rm(list=ls())

# load data
mydata = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_scores.csv")
head(mydata)

# make columns factors
mydata$Stim <- factor(ifelse(mydata$Stim == 0,"Continuous","ISF"), levels = c("Continuous", "ISF"))
mydata$Period <- factor(ifelse(mydata$Period == 0,"First","Second"), levels = c("First", "Second"))
mydata$Task <- factor(ifelse(mydata$Task == 1,"Pre","Post"), levels = c("Pre", "Post"))
mydata$ID <- as.factor(mydata$ID)
str(mydata)

# split data based on nback
d1 = mydata %>% filter(Nback == 1)
d2 = mydata %>% filter(Nback == 2)
d3 = mydata %>% filter(Nback == 3)
  
# evaluate performance change
m1 <- lmer(Score ~ Stim + Task + Period + Stim:Period + Stim:Task + (1 | ID) , data=d3, REML = FALSE)
anova(m1)
summary(m1)
coef(m1)

# evaluate reaction time change
m2 <- lmer(ResponsTime ~ Stim + Task + Period + Stim:Period + Stim:Task + (1|ID), data=d2, REML = FALSE)
anova(m2)
summary(m2)
print(m2)
plot(m2)
coef(m2)

emm <- emmeans(m2, ~Stim*Task)
contrast(emm)
