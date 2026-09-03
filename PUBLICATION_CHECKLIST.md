# Publication checklist

This copy is isolated from the working thesis and simulation repositories. Publishing it does not modify either source.

## Completed in this candidate

- Renamed the project around the research question rather than the misspelled working-repository name.
- Included the complete approved paper, LaTeX source, bibliography, and figures.
- Removed the author's personal ID number from the public paper source and rebuilt the PDF.
- Included aggregate welfare result tables.
- Included a curated simulation and validation subset, excluding the working database, cloud orchestration, archives, agent instructions, caches, and temporary notes.
- Replaced the machine-specific SUMO path with ordinary `SUMO_HOME` discovery.
- Added a public-facing overview that separates measured findings from limitations.

## Decisions required before public GitHub release

- Confirm supervisor/university expectations for sharing the paper and figures.
- Choose a code license. MIT is a reasonable default if no dependency or institutional restriction conflicts.
- Choose paper and data terms. A Creative Commons license may suit the paper, while aggregate results may need separate terms.
- Decide whether to publish only the aggregate evidence or also archive the raw simulation database with a DOI.
- Run a Git-history secret and personal-data scan after initializing the new repository.
- Add continuous integration for the database-independent tests.
- Add repository description, topics, social-preview image, and a tagged release.

## Suggested repository metadata

**Name:** `wardrop-routing-welfare-research`

**Description:** Welfare-based evaluation of backward-looking, forward-looking, and habitual routing under tolls and mixed adoption.

**Topics:** `transport-economics`, `operations-research`, `traffic-simulation`, `sumo`, `social-welfare`, `route-choice`, `congestion-pricing`
