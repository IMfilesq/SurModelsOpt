

processed = loader(raw.csv)
analysis = analyzer(processed, cnfg.boundaries)
plot = analysis.plot
stats = analysis.stats
