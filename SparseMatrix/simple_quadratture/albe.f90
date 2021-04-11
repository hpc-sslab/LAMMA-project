
  function al_dx(al,be,j,k,tmpnx,tmpny,stif) ! al part of Dx, where A = D+L+U
    use prec
    use common_var
    integer, intent(in):: j,k
    integer, intent(in):: tmpnx, tmpny
    real(mp), intent(in):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: al_dx
    al_dx =  (stif(1,1,j,k)+stif(3,3,j+1,k))*al(j,k)
  end function al_dx
  
  function al_lpux(al,be,j,k,tmpnx,tmpny,stif) ! al part of (L+U)x, where A = D+L+U
    use prec
    use common_var
    integer, intent(in):: j,k
    integer, intent(in):: tmpnx,tmpny
    real(mp), intent(in):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: al_lpux
    al_lpux = stif(1,3,j,k)*al(j-1,k) + stif(3,1,j+1,k)*al(j+1,k)&
         + stif(1,2,j,k)*be(j,k) + stif(1,4,j,k)*be(j,k-1) &
         + stif(3,2,j+1,k)*be(j+1,k) + stif(3,4,j+1,k)*be(j+1,k-1)
  end function al_lpux

  function be_dx(al,be,j,k,tmpnx,tmpny,stif) ! be part of Dx, where A = D+L+U
    use prec
    use common_var
    integer, intent(in):: j,k
    integer, intent(in):: tmpnx,tmpny
    real(mp), intent(in):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: be_dx
    be_dx =(stif(2,2,j,k)+stif(4,4,j,k+1))*be(j,k)
  end function be_dx
  
  function be_lpux(al,be,j,k,tmpnx,tmpny,stif) ! be part of (L+U)x, where A = D+L+U
    use prec
    use common_var
    integer, intent(in):: j,k
    integer, intent(in):: tmpnx,tmpny
    real(mp), intent(in):: al(0:tmpnx,1:tmpny), be(1:tmpnx,0:tmpny)
    real(mp), intent(in):: stif(1:4,1:4,1:tmpnx,1:tmpny)
    real(mp):: be_lpux
    be_lpux = stif(2,4,j,k)*be(j,k-1) + stif(4,2,j,k+1)*be(j,k+1) &
         + stif(2,1,j,k)*al(j,k) + stif(2,3,j,k)*al(j-1,k) &
         + stif(4,1,j,k+1)*al(j,k+1) + stif(4,3,j,k+1)*al(j-1,k+1)
  end function be_lpux

  
