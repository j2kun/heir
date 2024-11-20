"""Compute statistics about (non-Googler, non-bot) authors of commits."""

from collections import defaultdict
from datetime import datetime
import requests
import sys
import time

# Constants
GITHUB_API_URL = "https://api.github.com"
REPO = "google/heir"  # replace with the desired repo
# TOKEN = "your_github_token_here"  # Replace with your personal access token (if needed)

stats = []

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
    page = 0

    while url:
        print(f"Fetching page {page} of commits...")
        time.sleep(1)
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an error if the request failed
        data = response.json()
        commits.extend(data)

        # Check if there's another page of results
        if "next" in response.links:
            url = response.links["next"]["url"]
            page += 1
        else:
            url = None

    return commits


def extract_commits(commits):
    all_commits = []
    for commit in commits:
        commit_data = commit["commit"]
        author_data = commit_data.get("author", {})
        date_str = author_data.get("date")
        email = author_data.get("email")
        name = author_data.get("name")

        if not email or should_exclude(email):
            continue

        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        all_commits.append({
            "date": date,
            "author": name,
            "email": email,
            "message": commit_data.get("message").split("\n")[0]
        })

    return all_commits


def compute_stats(commits):
    monthly_authors = defaultdict(set)
    author_commit_count = defaultdict(int)
    monthly_commits = defaultdict(int)

    for commit in commits:
        name = commit["author"]
        date = commit["date"]
        year_month = date.strftime("%Y-%m")

        monthly_authors[year_month].add()
        monthly_commits[year_month] += 1
        author_commit_count[name] += 1

    stats = []
    for date, authors in monthly_authors.items():
        stats.append({
            "date": date,
            "distinct authors": len(authors),
            "commits": monthly_commits[date]
        })

    return stats


if __name__ == "__main__":
    try:
        commits = extract_commits(fetch_commits(REPO))
        stats = compute_stats(commits)
    except Exception as e:
        print(f"Error fetching commits: {e}")
        sys.exit(1)

    import csv
    with open("all_commits.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "author", "email", "message"])
        writer.writeheader()
        writer.writerows(commits)

    import csv
    with open("monthly_author_stats.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "distinct authors", "commits"])
        writer.writeheader()
        writer.writerows(stats)
