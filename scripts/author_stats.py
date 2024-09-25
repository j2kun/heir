"""Usage

gh api --paginate repos/google/heir/commits | jq -r '
  .[].commit
  | select(.author.email | contains("google.com") | not)
  | select(.author.email | contains("j2kun") | not)
  | select(.author.email | contains("kun.jeremy") | not)
  | select(.author.email | contains("dependabot") | not)
  | "\(.author.date), \(.author.email), \(.author.name)"
' | python scripts/author_stats.py
"""

from collections import defaultdict
from datetime import datetime
import sys


if __name__ == "__main__":
    commits = sys.stdin.read().strip().split("\n")

    monthly_authors = defaultdict(set)
    monthly_commits = defaultdict(int)
    total_author_commit_count = defaultdict(int)
    total_commits = 0

    for line in commits:
        date_str, email, author = [x.strip() for x in line.split(",")]
        date = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        year_month = date.strftime("%Y-%m")

        monthly_authors[year_month].add(author)
        monthly_commits[year_month] += 1
        total_author_commit_count[author] += 1
        total_commits += 1

    monthly_distinct_authors = {
        month: len(authors) for month, authors in monthly_authors.items()
    }

    print("Monthly distinct commit author count:")
    for month, count in monthly_distinct_authors.items():
        print(f"{month}: {count}")

    print("\nTotal commits by author:")
    for author, count in total_author_commit_count.items():
        print(f"{author}: {count}")

    print("\nTotal commits by month:")
    for month, count in monthly_commits.items():
        print(f"{month}: {count}")

    print(f"\nTotal commits: {total_commits}")
