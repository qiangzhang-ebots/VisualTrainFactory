#!/bin/bash
set -euo pipefail

MOUNTPOINT="/home/ebots/Desktop/DataHouse"
DEV="/dev/sda2"

if [[ ! -b "$DEV" ]]; then
  echo "错误: 找不到 $DEV"
  exit 1
fi

UUID=$(lsblk -no UUID "$DEV")
if [[ -z "$UUID" ]]; then
  echo "错误: 无法读取 $DEV 的 UUID"
  exit 1
fi

mkdir -p "$MOUNTPOINT"

sudo cp -a /etc/fstab "/etc/fstab.bak.$(date +%Y%m%d%H%M%S)"

TMP=$(mktemp)
grep -vE "(UUID=${UUID}|${MOUNTPOINT})" /etc/fstab > "$TMP" || true
printf '\n# Newsmy NTFS (/dev/sda2) -> Desktop/DataHouse\nUUID=%s  %s  ntfs-3g  defaults,uid=%s,gid=%s,umask=022,windows_names,locale=zh_CN.UTF-8,nofail  0  0\n' \
  "$UUID" "$MOUNTPOINT" "$(id -u)" "$(id -g)" >> "$TMP"
sudo cp "$TMP" /etc/fstab
rm -f "$TMP"

echo "已写入 /etc/fstab："
grep -n "DataHouse\|${UUID}" /etc/fstab

if findmnt "$MOUNTPOINT" >/dev/null; then
  sudo umount "$MOUNTPOINT"
fi
sudo mount "$MOUNTPOINT"

echo
echo "挂载成功："
df -h "$MOUNTPOINT"
findmnt "$MOUNTPOINT"
echo
echo "完成。重启后也会自动挂载到 Desktop/DataHouse。"
