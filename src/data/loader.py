

processed = loader(raw.csv)
analysis = reporter(processed, cnfg.boundaries)
plot = analysis.plot
stats = analysis.stats
