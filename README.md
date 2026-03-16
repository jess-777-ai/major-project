INFO: *.csv and *.npz files are NOT uploaded since they too BIG!

=== TO UPDATE NEW CHANGES TO YOUR GITHUB REPO

1. Open the path to your project on Powershell

2. 
git status
git add .
git commit -m "Describe what changed"
git push

=== Very important rule for your project

Since you're working with large ML datasets, never push:

.csv
.npz
.pth checkpoints (unless small)

Your .gitignore should already block them (exists as a hidden file on your local project folder)
