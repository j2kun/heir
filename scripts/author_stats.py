"""Compute statistics about (non-Googler, non-bot) authors of commits."""

import requests
import sys
from collections import defaultdict
from datetime import datetime

# Constants
GITHUB_API_URL = "https://api.github.com"
REPO = "google/heir"  # replace with the desired repo
# TOKEN = "your_github_token_here"  # Replace with your personal access token (if needed)

monthly_authors = defaultdict(set)
author_commit_count = defaultdict(int)
monthly_commits = defaultdict(int)
total_commits = 0

# Define email patterns to exclude
excluded_patterns = ["google.com", "j2kun", "kun.jeremy", "dependabot"]

# Function to check if an email should be excluded
def should_exclude(email):
    return any(pattern in email for pattern in excluded_patterns)

# Function to fetch all commits from a repository (with pagination)
def fetch_commits(repo):
    url = f"{GITHUB_API_URL}/repos/{repo}/commits"
    # headers = {"Authorization": f"token {TOKEN}"}
    params = {"per_page": 100}  # Fetch up to 100 commits per page
    commits = []

    while url:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error if the request failed
        data = response.json()
        commits.extend(data)

        # Check if there's another page of results
        if "next" in response.links:
            url = response.links["next"]["url"]
        else:
            url = None

    return commits

# Function to process commits
def process_commits(commits):
    global total_commits
    global monthly_commits
    for commit in commits:
        commit_data = commit["commit"]
        author_data = commit_data.get("author", {})
        date_str = author_data.get("date")
        email = author_data.get("email")
        name = author_data.get("name")

        if not email or should_exclude(email):
            continue

        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        year_month = date.strftime("%Y-%m")  # e.g., '2024-02'

        monthly_authors[year_month].add(name)
        monthly_commits[year_month] += 1
        author_commit_count[name] += 1
        total_commits += 1


if __name__ == "__main__":
    try:
        commits = fetch_commits(REPO)
        process_commits(commits)
    except Exception as e:
        print(f"Error fetching commits: {e}")
        sys.exit(1)

    monthly_distinct_authors = {
        month: len(authors) for month, authors in monthly_authors.items()
    }

    print("Monthly distinct commit author count:")
    for month, count in monthly_distinct_authors.items():
        print(f"{month}: {count}")

    print("\nTotal commits by author:")
    for author, count in author_commit_count.items():
        print(f"{author}: {count}")

    print("\nTotal commits by month:")
    for month, count in monthly_commits.items():
        print(f"{month}: {count}")

    print(f"\nTotal commits: {total_commits}")
