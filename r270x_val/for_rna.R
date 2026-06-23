library(tidyverse)
library(ggplot2)
library(ggprism)
library(scales)

# Usage:
#   source("plot_eon_editing_rates.R")
#   main("r270x_z_eons.csv")

cols <- c(
  neg       = "#7A9FFF",
  pos       = "#F47C7C",
  curve     = "#FFC300",
  pr        = "#c9aaee",
  threshold = "#AD7800",
  diagonal  = "grey55",
  teal      = "#7FA2AC"
)

theme_binary <- function() {
  theme_prism(
    palette       = "winter_bright",
    base_size     = 12,
    base_family   = "sans",
    base_fontface = "plain",
    base_line_size = 1,
    base_rect_size = 1,
    border        = FALSE
  ) +
    theme(
      plot.title    = element_text(face = "bold", size = 16, margin = margin(b = 2)),
      plot.subtitle = element_text(margin = margin(b = 4)),
      axis.title    = element_text(face = "bold"),
      legend.title  = element_text(face = "bold", size = 12),
      legend.position = "top"
    )
}

main <- function(csv_file) {
  df <- read_csv(csv_file, show_col_types = FALSE) %>%
    mutate(
      id   = if ("id" %in% names(.)) id else row_number(),
      mean = as.numeric(mean),
      SD   = as.numeric(.data$SD)
    ) %>%
    arrange(id)
  
  # Normalise to [0,1] if values look like percentages
  if (max(df$mean, na.rm = TRUE) > 1) {
    df <- df %>% mutate(
      mean = mean / 100,
      SD   = SD   / 100
    )
  }
  
  eon_plot <- ggplot(df, aes(x = id, y = mean)) +
    geom_col(fill = cols[["teal"]], alpha = 0.80, width = 0.7) +
    geom_errorbar(
      aes(ymin = mean - SD, ymax = mean + SD),
      width     = 0.25,
      linewidth = 0.6,
      color     = cols[["diagonal"]]
    ) +
    scale_x_discrete() +
    scale_y_continuous(
      labels = percent_format(accuracy = 1),
      expand = expansion(mult = c(0, 0.02))
    ) +
    coord_cartesian(ylim = c(0, 1)) +
    labs(
      title    = "EON Editing Rates",
      subtitle = "DLA, Actin B mRNA, n=3 per EON",
      x        = "EON ID",
      y        = "Mean editing level"
    ) +
    theme_binary() +
    theme(
      legend.position = "none",
      axis.text.x  = element_text(angle = 45, hjust = 1),
      axis.title.x = element_text(margin = margin(t = 12))
    )
  
  print(eon_plot)
  invisible(eon_plot)
}

main("r270x_z_eons.csv")