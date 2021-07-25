# ml-models
A repository for stock implementations of popular machine learning models.

# Design Style

Every model should at mininum
- Have separate scripts for training and analyzing models before production
- Use a commandline interface.

Use pipreqs to generate requirements.txt
Almost always use the latest packages (no rollbacks).

# Notes

I'm terrible at Flask, so help me. Issues or PRs welcome.

# TODO
Add pandas_profiling to the analysis.
Learn Flask or FastAPI -- any deployable method.
Is a package format really necessary to use these? Maybe it should be a cookie-cutter