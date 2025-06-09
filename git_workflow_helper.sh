#!/bin/bash
# Git Workflow Helper Script
# Run this periodically to stay synchronized with your team

echo "🔄 Fetching latest changes from remote..."
git fetch

echo "📊 Checking branch status..."
git status --porcelain

echo "🔍 Checking for new commits on main..."
BEHIND=$(git rev-list --count HEAD..origin/main 2>/dev/null || echo "0")
AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "0")

if [ "$BEHIND" -gt 0 ]; then
    echo "⚠️  Your branch is $BEHIND commit(s) behind origin/main"
    echo "📋 New commits you don't have:"
    git log --oneline --graph HEAD..origin/main
fi

if [ "$AHEAD" -gt 0 ]; then
    echo "⏫ Your branch is $AHEAD commit(s) ahead of origin/main"
fi

if [ "$BEHIND" -eq 0 ] && [ "$AHEAD" -eq 0 ]; then
    echo "✅ Your branch is up to date with origin/main"
fi

echo "🌿 Available remote branches:"
git branch -r --sort=-committerdate | head -10

echo "📈 Recent activity:"
git log --oneline --graph --all -10
