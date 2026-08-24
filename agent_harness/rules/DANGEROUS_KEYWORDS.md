> 危险命令关键词，安全预检用

rm -rf
shred
unlink
truncate
dd
mkfs
fdisk
parted
mkswap
mount
umount
> /dev/sd*
chmod
chown
chattr
setfacl
shutdown
reboot
halt
poweroff
init
kill -9
killall
pkill
iptables
ufw
systemctl
update-rc.d
ifconfig
ip link
route
ip route
nmcli
git push --force
git reset --hard
git clean -fd
git branch -D
git rebase
git merge --abort
git reflog delete
docker rm -f
docker rmi -f
docker system prune -af
docker volume rm
docker network rm
docker run --privileged
docker exec
DROP TABLE
DROP DATABASE
DELETE FROM
UPDATE ... SET
TRUNCATE TABLE
ALTER TABLE ... DROP
ALTER DATABASE
REINDEX
VACUUM FULL
apt purge
dpkg -P
pip uninstall
npm uninstall
gem uninstall
cargo remove
brew remove
>
mv
cp -f
ln -sf
install
:(){ :|:& };:
while true
xargs
find ... -exec
eval
source
renice
ulimit
