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
library(stringr)

# clear enviornment of variables
rm(list=ls())

# load data
mydata_2obs = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_em_diff")
mydata_4obs = read.csv("/Volumes/GoogleDrive/.shortcut-targets-by-id/1WuuFja-yoluAKvFp--yOQe7bKLg-JeA-/EMOTIONLINE/MastersThesis/DataProcessing/EmotionModel/src/assets/df_em_4obs")
str(mydata_2obs)
str(mydata_4obs)

# make columns factors
mydata_2obs$ID <- as.factor(mydata_2obs$ID)
mydata_4obs$ID <- as.factor(mydata_4obs$ID)
mydata_2obs$Nback <- as.factor(mydata_2obs$Nback)
mydata_4obs$Nback <- as.factor(mydata_4obs$Nback)
mydata_2obs$Period <- as.factor(mydata_2obs$Period)
mydata_4obs$Period <- as.factor(mydata_4obs$Period)
mydata_4obs$Task <- factor(ifelse(mydata_4obs$Task == 1,"Pre","Post"), levels = c("Pre", "Post"))
mydata_2obs$Stim <- factor(ifelse(mydata_2obs$Stim == 1,"Continuous","ISF"), levels = c("Continuous", "ISF"))
mydata_4obs$comb <- paste(mydata_4obs[,"ID"],mydata_4obs[,"Nback"])
mydata_2obs$comb <- paste(mydata_2obs[,"ID"],mydata_2obs[,"Nback"])
str(mydata_2obs)

# split data based on nback
d1_2obs = mydata_2obs %>% filter(Nback == 1)
d2_2obs = mydata_2obs %>% filter(Nback == 2)
d3_2obs = mydata_2obs %>% filter(Nback == 3)
d1_4obs = mydata_4obs %>% filter(Nback == 1)
d2_4obs = mydata_4obs %>% filter(Nback == 2)
d3_4obs = mydata_4obs %>% filter(Nback == 3)

# filter for plotting
mydata_4obs_stim0 = mydata_4obs %>% filter(mydata_4obs$Stim == 0)
mydata_4obs_stim1 = mydata_4obs %>% filter(mydata_4obs$Stim == 1)
d3_4obs_stim0 = d3_4obs %>% filter(d3_4obs$Stim == 0)
d3_4obs_stim1 = d3_4obs %>% filter(d3_4obs$Stim == 1)
d2_4obs_stim0 = d2_4obs %>% filter(d2_4obs$Stim == 0)
d2_4obs_stim1 = d2_4obs %>% filter(d2_4obs$Stim == 1)
d1_4obs_stim0 = d1_4obs %>% filter(d1_4obs$Stim == 0)
d1_4obs_stim1 = d1_4obs %>% filter(d1_4obs$Stim == 1)
str(mydata_4obs_stim0)

# Make theme
mytheme <- theme(text = element_text(size=20), 
                 axis.title.y = element_text(angle=90),
                 legend.position = c(0.25,0.1), legend.key = element_blank(), legend.direction = "horizontal",
                 legend.background = element_rect(size=0.5, linetype = "solid", colour = "black"),
                 plot.title = element_text(size=25, hjust=0.5, face="bold"))

emotion_list <- list("happiness", "sadness", "disgust", "surprise", "anger", "fear")

for (emotion in emotion_list){
# spagetti plot
p1 <- ggplot(data = mydata_4obs_stim0, aes_string(x = "Task", y = emotion, group = "comb", colour="Nback")) +
  mytheme + 
  #coord_trans(ylim = c(0.3,0.75))  + 
  geom_line(size=1.5) + 
  ggtitle("Continuous") +
  scale_color_manual(values=c("#2a9df4", "#1167b1","#003d80")) +
  labs(y = paste("frames with ",emotion), x = emotion) + 
  scale_x_discrete(limits = c("Pre","Post"), expand = c(0.1, 0.1))

p2 <- ggplot(data = mydata_4obs_stim1, aes_string(x = "Task", y = emotion, group = "comb", colour="Nback")) +
  mytheme + 
  #coord_trans(ylim = c(0.3,0.75))  +  
  geom_line(size=1.5) + 
  ggtitle("ISF") +
  labs(x = emotion) + 
  scale_color_manual(values=c("#2a9df4", "#1167b1","#003d80")) +
  labs(y = paste("frames with ",emotion), x = "Stimulation") + 
  scale_x_discrete(limits = c("Pre","Post"), expand = c(0.1, 0.1))

grid.arrange(p1,p2, ncol=2)
}

mytheme <- theme(text = element_text(size=20), 
                 axis.title.y = element_text(angle=90),
                 legend.position = c(0.5,0.15), legend.key = element_blank(), legend.direction = "vertical",
                 legend.background = element_rect(size=0.5, linetype = "solid", colour = "black"),
                 plot.title = element_text(size=25, hjust=0.5, face="bold"))

for (emotion in emotion_list){
# spagetti plot
p3 <- ggplot(data = mydata_2obs, aes_string(x = "Stim", y = emotion, fill="Stim")) +
  mytheme + 
  geom_boxplot(width=0.7, alpha=0.6) +
  geom_dotplot(binaxis='y', stackdir='center', dotsize=0.8, alpha=0.7) +
  ggtitle(paste("Average Change in", toupper(emotion)," Before and After Stimulation")) +
  guides(fill= guide_legend(reverse = FALSE)) +
  scale_fill_manual(values=c("#398BED", "#F3B532")) +
  labs(y = "counts", x = "Stimulation")

grid.arrange(p3)
}
