# Upload through the GitHub website

## Recommended repository settings

- **Repository name:** `paper2-global-spectral-manifold-complexity`
- **Description:** `Code, frozen results, figures, and manuscript for cross-architecture spectral-manifold complexity and neural-surrogate learning difficulty in nanophotonics.`
- **Visibility:** Private until the author metadata, data DOI, coauthor approvals, and licensing decisions in `RELEASE_CHECKLIST.md` are complete.
- **Initialize with README:** Leave unchecked because this package already contains a README, license, and `.gitignore`.

## Upload

1. Sign in to GitHub and select **New repository**.
2. Enter the recommended repository name and description.
3. Choose **Private** for the first upload.
4. Create the empty repository.
5. Select **uploading an existing file**.
6. Unzip the GitHub package on your computer.
7. Open the unzipped `paper2-global-spectral-manifold-complexity` folder and drag **its contents** into the upload area. Do not upload the outer folder as one nested directory.
8. Use this commit message:

   ```text
   Initial reproducibility release for Paper 2
   ```

9. Select **Commit changes**.
10. Open the **Actions** tab and confirm that `repository-validation` passes.

GitHub's web uploader may reject a very large batch or hide dotfiles in some operating-system file pickers. If `.github`, `.gitignore`, or `.gitattributes` is missing after the first upload, upload those items separately. Do not upload anything from `data/raw/`.

## Before changing visibility to public

Complete every unresolved item in `docs/RELEASE_CHECKLIST.md`. The current package is suitable for private collaboration and technical review, but it is not ready for a public archival release while the manuscript metadata, rights, raw-data location, and 20-layer provenance discrepancy remain unresolved.

