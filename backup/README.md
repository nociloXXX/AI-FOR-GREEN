# Backup module

Automatic backup and sync for the project repository.

## What it does

- Auto-commits changed files and pushes them to the Git remote on a timer.
- Mirrors every commit to a local bare repository on an external drive.
- Takes a full working-tree snapshot on a separate timer.
- Appends a timestamped activity log for commits, uploads, snapshots, and errors.

## Configuration

- `auto_sync.ps1 -Drive <drive>` selects the backup drive. The default is `F:\`.
- The project directory is detected automatically from the script location.

## Scheduled tasks (Windows)

Create two tasks that point at `auto_sync.ps1`:

- `AI-FOR-GREEN AutoSync`: every 5 minutes, run `-Mode Sync`.
- `AI-FOR-GREEN Snapshot`: every 30 minutes, run `-Mode Snapshot`.

## Paths created

- `<drive>:\AI-FOR-GREEN_backup` - bare git mirror
- `<drive>:\AI-FOR-GREEN_logs\activity.log` - activity log
- `<drive>:\AI-FOR-GREEN_snapshots\<timestamp>\` - timed snapshots
- `<project>\.autosync\activity.log` - local fallback log

## Troubleshooting

1. Read the activity log first.
2. If the drive is missing, read `<project>\.autosync\activity.log`.
3. Check task state with `Get-ScheduledTask -TaskName 'AI-FOR-GREEN*'`.
4. Restore with `git clone <drive>:\AI-FOR-GREEN_backup`.
