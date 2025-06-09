# Git Workflow Quick Commands
# Run these periodically to stay synchronized

# Basic sync check
git fetch && git status

# See if main branch has new commits
git fetch && git log --oneline HEAD..origin/main

# See your unpushed commits
git log --oneline origin/main..HEAD

# Check all branch activity
git fetch && git log --oneline --graph --all -10

# See who's been working on what
git fetch && git log --oneline --all --since="1 week ago" --author-date-order

# Quick branch status overview
git fetch && git branch -vv
