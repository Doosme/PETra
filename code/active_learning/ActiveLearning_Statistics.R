library(ggplot2)
library(tidyr)
library(dplyr)

# MONOLINGUAL

# Data
df <- data.frame(
  Metric = c("DE-ACC", "DE-F1", "ES-ACC", "ES-F1"),
  L0 = c(0.8089, 0.0564, 0.7194, 0.3076),
  L8 = c(0.8690, 0.5490, 0.8041, 0.4861),
  L13 = c(0.8779, 0.5732, 0.8096, 0.5921)
)

df_long <- pivot_longer(df, cols = c(L0, L8, L13),
                        names_to = "Layer", values_to = "Value")

df_long$Layer <- factor(df_long$Layer, levels = c("L0", "L8", "L13"))

df_long <- df_long %>%
  mutate(
    MetricType = ifelse(grepl("ACC", Metric), "Accuracy", "F1-Score"),
    LegendGroup = sub("-.*", "", Metric)
  )

# Plot
ggplot(df_long, aes(x = Layer, y = Value,
                    color = LegendGroup, group = Metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2, shape = 21, fill = "white", stroke = 1) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = c(0, 0)
  ) +
  facet_wrap(~MetricType, ncol = 2) +
  scale_color_manual(values = c(
    "DE" = "#D55E00",
    "ES" = "#0072B2"
  )) +
  labs(
    x = "Active Learning Loop",
    y = "Performance",
    color = "Language"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8)
  )



#CROSSLINGUAL
# Data
df <- data.frame(
  Metric = c("PT-ACC", "PT-F1", "IT-ACC", "IT-F1", "NL-ACC", "NL-F1",
             "EL-ACC", "EL-F1", "SV-ACC","SV-F1", "FR-ACC", "FR-F1"),
  L0 = c(0.60, 0.46, 0.64, 0.39, 0.63, 0.58, 0.51, 0.19, 0.59, 0.39, 0.59, 0.46),
  L8 = c(0.76, 0.70, 0.80, 0.77, 0.73, 0.67, 0.67, 0.68, 0.68, 0.50, 0.81, 0.81),
  L13 = c(0.77, 0.72, 0.83, 0.80, 0.72, 0.65, 0.71, 0.73, 0.74, 0.63, 0.75, 0.73),
  L14 = c(0.82, 0.80, 0.84, 0.82, 0.83, 0.82, 0.74, 0.76, 0.77, 0.68, 0.84, 0.84)
)

df_long <- pivot_longer(df, cols = c(L0, L8, L13, L14),
                        names_to = "Layer", values_to = "Value")

df_long$Layer <- factor(df_long$Layer, levels = c("L0", "L8", "L13", "L14"))

df_long <- df_long %>%
  mutate(
    MetricType = ifelse(grepl("ACC", Metric), "Accuracy", "F1-Score"),
    LegendGroup = sub("-.*", "", Metric)
  )

# Plot
ggplot(df_long, aes(x = Layer, y = Value,
                    color = LegendGroup, group = Metric)) +
  geom_line(linewidth = 1) +
  geom_point(size = 2, shape = 21, fill = "white", stroke = 1) +
  scale_y_continuous(
    limits = c(0, 1),
    expand = c(0, 0)
  ) +
  facet_wrap(~MetricType, ncol = 2) +
  scale_color_manual(values = c(
    "PT" = "#D55E00",
    "IT" = "#009E73",
    "FR" = "#0072B2",
    "NL" = "#E69F00",
    "SV" = "#56B4E9",
    "EL" = "#CC79A7"
  )) +
  labs(
    x = "Active Learning Loop",
    y = "Performance",
    color = "Language"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    axis.title = element_text(face = "bold"),
    strip.text = element_text(face = "bold"),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.border = element_rect(color = "black", fill = NA, linewidth = 0.8)
  )