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
mydata = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_collective")

#group <- rep(rep(c("A", "A", "B", "A", "A", "B", "B", "B", "A", "B"), each=2), 3)
#mydata["group"] = group
head(mydata)

# make columns factors
mydata$Stim <- as.factor(mydata$Stim)
mydata$Task <- factor(ifelse(mydata$Task == 2,"Post","Pre"), levels = c("Pre", "Post"))
mydata$ID <- as.factor(mydata$ID)
str(mydata)

# split data based on nback
d1 = mydata %>% filter(Nback == 1)
d2 = mydata %>% filter(Nback == 2)
d3 = mydata %>% filter(Nback == 3)

# Set up 
m <- glmer(AU12 ~ Stim*Session + (1 | ID), data=d3, family="binomial")
summary(m)

# Perform post-hoc test 
emm <- emmeans(m, "Stim")
pairs(emm)
summary(glht(m, linfct = mcp(Stim = "Tukey"), alternative = c("greater"), df=16), test = adjusted("fdr"))

# plot model 
p <- plot_model(m)
p + theme_sjplot()

#
d3_stim0 = d3 %>% filter(d3$Stim == 0)  
d3_stim0$Task <-  factor(d3_stim0$Task, levels = c("Pre","Post"))
d3_stim1 = d3 %>% filter(d3$Stim == 1) 
d3_stim1$Task <- factor(d3_stim1$Task, levels = c("Pre","Post"))

stim0_pre <- d3_stim0$AU12[d3_stim0$Task == "Pre"]
stim0_post <- d3_stim0$AU12[d3_stim0$Task == "Post"]
stim1_pre <- d3_stim1$AU12[d3_stim1$Task == "Pre"]
stim1_post <- d3_stim1$AU12[d3_stim1$Task == "Post"]

# Make theme
mytheme <- theme(text = element_text(size=20), 
                 axis.title.y = element_text(angle=90),
                 legend.position = c(0.5,0.5), legend.key = element_blank(),
                 legend.background = element_rect(size=0.5, linetype = "solid", colour = "black"),
                 plot.title = element_text(size=25, hjust=0.5, face="bold"))

# Side
p1 <- ggplot(data = d3_stim0, aes(x = AU12, fill = Task)) +
  mytheme + 
  ggtitle("Continuous") +
  geom_density(adjust=1, alpha=.4, aes(x=AU12, y = -..density..), subset(d3_stim0, Task == "Pre") ) +
  geom_density(adjust=1, alpha=.4, aes(x=AU12, y = ..density..), subset(d3_stim0, Task == "Post") ) +
  coord_flip(ylim = c(-30,30)) +
  scale_color_manual(values=c("#00a8e8","#007ea7")) +
  labs(y = "Magnitude", x = "Logits") +
  guides(fill = guide_legend(reverse = TRUE, direction="horizontal")) +
  scale_x_continuous(breaks=seq(0,1))
  #annotate("text", x=0.93, y=1, label= "87%", size=10, color="white") + 
  #annotate("text", x=0.07, y=1, label= "13%", size=10, color="black") +
  #annotate("text", x=0.93, y=-1, label= "87%", size=10, color="white") +
  #annotate("text", x=0.07, y=-1.1, label= "13%", size=10, color="black")

p2 <- ggplot(data = d3_stim1, aes(x = AU12, fill = Task)) +
  mytheme + 
  ggtitle("ISF") +
  labs(x = "Stimulation") + 
  geom_density(adjust=1, alpha=.4, aes(x=AU12, y = -..density..), subset(d3_stim1, Task == "Pre") ) +
  geom_density(adjust=1, alpha=.4, aes(x=AU12, y = ..density..), subset(d3_stim1, Task == "Post") ) +
  coord_flip(ylim = c(-30,30)) + 
  scale_color_manual(values=c("#007ea7", "#00a8e8")) +
  labs(y = "Magnitude", x = "Logits") +
  guides(fill = guide_legend(reverse = TRUE, direction="horizontal")) +
  scale_x_continuous(breaks=seq(0,1))
  #annotate("text", x=0.93, y=1, label= "89%", size=10, color="white") + 
  #annotate("text", x=0.07, y=1, label= "11%", size=10, color="black") +
  #annotate("text", x=0.93, y=-1, label= "83%", size=10, color="white") +
  #annotate("text", x=0.07, y=-1.2, label= "17%", size=10, color="black")

grid.arrange(p1,p2, ncol=2)



