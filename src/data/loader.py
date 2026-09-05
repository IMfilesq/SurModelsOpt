

processed = loader("data/raw.csv")
analysis = reporter(processed, cfg.boundaries)
plot = analysis.plot
stats = analysis.stats
itp
