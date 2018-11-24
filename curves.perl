#!/usr/bin/perl

while (<>){
    if (/Epoch (\d+) TRAIN loss=([^ ]+) \((.*)\) lr=([^ ]+) /){
	$epoch = $1;
	$loss = $2;
	($a,$p,$r,$f)= split /\,/,$3;
	$lr=$4;
	$f =~ s/F//;
	$TF{$epoch}=$f;
	$TL{$epoch}=$loss;
	$TF{$epoch}=$f;
	$lr{$epoch}=$lr;
    }
    if (/Epoch (\d+) VALID loss=([^ ]+) \((.*)\) /){
	$epoch = $1;
	$loss = $2;
	($a,$p,$r,$f)= split /\,/,$3;
	$f =~ s/F//;
	$VF{$epoch}=$f;
	$VL{$epoch}=$loss;
    }
}

print "Epoch\tT_Fscr\tT__loss\tV_Fscr\tV__loss\tLR\n";
foreach $epoch (sort {$a <=> $b} keys %TF){
    print "$epoch\t$TF{$epoch}\t$TL{$epoch}\t$VF{$epoch}\t$VL{$epoch}\t$lr{$epoch}\n";
}
