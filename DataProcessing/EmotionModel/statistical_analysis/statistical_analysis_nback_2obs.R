# installs
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

# package enviornment of variables
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

# clear shit
rm(list=ls())

# load data
mydata = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_big.csv")

group <- rep(rep(c("A", "A", "B", "A", "A", "B", "B", "B", "A", "B"),each=1), 6)
mydata["group"] = group
head(mydata)

# make columns factors
mydata$Period <- as.factor(mydata$Period)
#mydata$Stim <- as.factor(mydata$Stim)
mydata$Stim <- factor(ifelse(mydata$Stim == 1,"ISF","Continuous"), levels = c("ISF", "Continuous"))
#mydata$Task <- factor(ifelse(mydata$Task == 1,"Pre","Post"), levels = c("Pre", "Post"))
mydata$ID <- as.factor(mydata$ID)
str(mydata)

# split data based on nback
d1 = mydata %>% filter(Nback == 1)
d2 = mydata %>% filter(Nback == 2)
d3 = mydata %>% filter(Nback == 3)
str(d3)

# Set up 
m <- lmer(Score ~ Stim*Period + ( 1 | ID), data=d3)
#m <- lme(Score ~ Stim*Period, data=d3, random = ~1|ID)
summary(m)
anova(m)
step(m, reduce.fixed=FALSE, reduce.random = FALSE)

# Perform post-hoc test 
emm <- emmeans(m, "Stim")
pwpm(emm)
eff_size(emm, sigma = sigma(m), edf = Inf)

summary(glht(m, linfct = mcp(Stim = "Tukey"), alternative = c("less"), df=16), test = adjusted("fdr"))

# Plot a t-distribution
pt = dist_t(
  #t = NULL,
  #main = "title",
  deg.f = 18,
  p = 0.05,
  xmax = 3.5,
  #geom.colors = NULL,
  geom.alpha = 0.7
  #title="t-distribution",
)
pt.add_title("t-distribution")
plot(pt)

# plot model 
p <- plot_model(m)
p + theme_sjplot()

# Make theme
mytheme <- theme(text = element_text(size=20), 
                 axis.title.y = element_text(angle=90),
                 legend.position = c(0.25,0.1), legend.key = element_blank(), legend.direction = "horizontal",
                 legend.background = element_rect(size=0.5, linetype = "solid", colour = "black"),
                 plot.title = element_text(size=25, hjust=0.5, face="bold"))

# spagetti plot
p <- ggplot(data = d3, aes(x = Stim, y = Score, fill=Stim)) +
  mytheme + 
  geom_violin(trim=FALSE) + 
  geom_boxplot(width=0.1, fill="white") +
  ggtitle("Continuous") +
  #coord_trans(ylim = c(73,100))  + 
  guides(fill= guide_legend(reverse = TRUE)) +
  scale_color_manual(values=c("#007ea7", "#00a8e8")) +
  labs(y = "Percent Accuracy [%]", x = "Stimulation") 
  #scale_x_discrete(limits = c("Pre","Post"), expand = c(0.1, 0.1))

p
str(d3)
