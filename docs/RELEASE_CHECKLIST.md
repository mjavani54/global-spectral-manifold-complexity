# Public-release checklist

The repository is technically organized, but the following items are still required before making it public or minting a DOI:

- [ ] Replace `[COAUTHORS]`, `[Additional affiliations]`, `[email]`, and the funding placeholder in the manuscript.
- [ ] Add every confirmed author to `CITATION.cff` in the correct order.
- [ ] Confirm that all coauthors approve public release of the manuscript, figures, code, and frozen results.
- [ ] Choose and document licenses for figures, results, and any deposited data.
- [ ] Deposit the raw data archive and add its DOI/access terms.
- [ ] Resolve or explicitly accept the 20-layer master-summary mismatch.
- [ ] Add the GitHub repository URL to `CITATION.cff` after the repository exists.
- [ ] Add the article DOI, year, volume, and pages after publication.
- [ ] Run `python scripts/validate_release.py`.
- [ ] Confirm that no proprietary Lumerical files, credentials, or institutional secrets are present.
- [ ] Create a tagged release (recommended: `v1.0.0`) only after the items above are complete.

